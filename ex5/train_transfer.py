import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_datasets(cfg):
    ds_cfg = cfg['dataset']
    img_size = tuple(ds_cfg['image_size'])

    if ds_cfg['format'] != 'directory':
        raise ValueError('Only directory dataset format is supported currently.')

    train_ds = keras.preprocessing.image_dataset_from_directory(
        ds_cfg['path'],
        validation_split=ds_cfg['validation_split'],
        subset='training',
        seed=ds_cfg['seed'],
        image_size=img_size,
        batch_size=cfg['training']['batch_size'],
        label_mode=ds_cfg['class_mode']
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        ds_cfg['path'],
        validation_split=ds_cfg['validation_split'],
        subset='validation',
        seed=ds_cfg['seed'],
        image_size=img_size,
        batch_size=cfg['training']['batch_size'],
        label_mode=ds_cfg['class_mode']
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def build_augmentation(cfg):
    aug_cfg = cfg['augmentation']
    if not aug_cfg.get('enabled', False):
        return keras.Sequential(name='no_aug')

    layers_list = []
    if aug_cfg.get('horizontal_flip'): layers_list.append(layers.RandomFlip('horizontal'))
    if aug_cfg.get('rotation_range'): layers_list.append(layers.RandomRotation(aug_cfg['rotation_range']/360.0))
    if aug_cfg.get('zoom_range'): layers_list.append(layers.RandomZoom(aug_cfg['zoom_range']))
    if aug_cfg.get('width_shift_range') or aug_cfg.get('height_shift_range'):
        layers_list.append(layers.RandomTranslation(
            height_factor=aug_cfg.get('height_shift_range', 0),
            width_factor=aug_cfg.get('width_shift_range', 0)))
    return keras.Sequential(layers_list, name='augmentation')


def build_model(cfg, num_classes):
    model_cfg = cfg['model']
    head_cfg = cfg['head']

    base = getattr(keras.applications, model_cfg['base_architecture'])(
        include_top=model_cfg['include_top'],
        weights=model_cfg['weights'],
        input_shape=(*cfg['dataset']['image_size'], 3)
    )

    if model_cfg['trainable_strategy'] in ('freeze_then_finetune', 'freeze'):
        base.trainable = False
    else:
        base.trainable = True

    inputs = keras.Input(shape=(*cfg['dataset']['image_size'], 3))
    x = build_augmentation(cfg)(inputs)
    prep_fn = getattr(keras.applications, model_cfg['base_architecture']).preprocess_input
    x = layers.Lambda(prep_fn)(x)
    x = base(x, training=False)

    if head_cfg['global_pool'] == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif head_cfg['global_pool'] == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    else:
        x = layers.Flatten()(x)

    for units in head_cfg['dense_units']:
        x = layers.Dense(units, activation=head_cfg['activation'])(x)
        if head_cfg.get('dropout', 0) > 0:
            x = layers.Dropout(head_cfg['dropout'])(x)

    outputs = layers.Dense(head_cfg['output_classes'], activation=head_cfg['output_activation'])(x)
    model = keras.Model(inputs, outputs)
    return model, base


def compile_model(model, lr, optimizer_name):
    if optimizer_name == 'adam':
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError('Unsupported optimizer')

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def train_feature_extraction(model, cfg, train_ds, val_ds):
    tr_cfg = cfg['training']
    callbacks = []

    if tr_cfg.get('early_stopping'):
        es_cfg = tr_cfg['early_stopping']
        callbacks.append(keras.callbacks.EarlyStopping(monitor=es_cfg['monitor'], patience=es_cfg['patience'], restore_best_weights=True))

    if tr_cfg.get('checkpoint'):
        ck_cfg = tr_cfg['checkpoint']
        Path('checkpoints').mkdir(exist_ok=True)
        callbacks.append(keras.callbacks.ModelCheckpoint('checkpoints/best.keras', monitor=ck_cfg['monitor'], save_best_only=ck_cfg['save_best_only']))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=tr_cfg['feature_extraction_epochs'],
        callbacks=callbacks
    )
    return history


def fine_tune(model, base, cfg, train_ds, val_ds):
    model_cfg = cfg['model']
    tr_cfg = cfg['training']
    if model_cfg['trainable_strategy'] != 'freeze_then_finetune':
        return None

    # Unfreeze last N layers
    for layer in base.layers[-model_cfg['fine_tune_unfreeze_layers']:]:
        layer.trainable = True

    compile_model(model, tr_cfg['learning_rates']['fine_tune'], tr_cfg['optimizer'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=tr_cfg['fine_tune_epochs']
    )
    return history


def evaluate_and_report(model, val_ds, class_names, cfg):
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    report_cfg = cfg['outputs']

    if report_cfg.get('confusion_matrix'):
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:\n', cm)
    if report_cfg.get('classification_report'):
        print('Classification Report:\n', classification_report(y_true, y_pred, target_names=class_names))

    metrics_cfg = cfg.get('metrics', [])
    if 'precision' in metrics_cfg:
        print('Precision:', precision_score(y_true, y_pred, average='macro'))
    if 'recall' in metrics_cfg:
        print('Recall:', recall_score(y_true, y_pred, average='macro'))
    if 'f1' in metrics_cfg:
        print('F1:', f1_score(y_true, y_pred, average='macro'))


def main():
    cfg = load_config('config_transfer.json')

    train_ds, val_ds, class_names = build_datasets(cfg)

    model, base = build_model(cfg, len(class_names))
    compile_model(model, cfg['training']['learning_rates']['feature_extraction'], cfg['training']['optimizer'])

    print('Starting feature extraction phase...')
    hist1 = train_feature_extraction(model, cfg, train_ds, val_ds)

    print('Starting fine-tuning phase (if configured)...')
    hist2 = fine_tune(model, base, cfg, train_ds, val_ds)

    evaluate_and_report(model, val_ds, class_names, cfg)

    out_cfg = cfg['outputs']
    if out_cfg.get('export_saved_model'):
        export_path = out_cfg['export_saved_model']
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(export_path)
        print(f'Model saved to {export_path}')


if __name__ == '__main__':
    main()
