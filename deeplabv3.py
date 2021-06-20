import tensorflow as tf
import numpy as np

def vgg16_backbone(img_shape):
    original = tf.keras.applications.VGG16(include_top=False, \
                                     weights='imagenet', \
                                     input_tensor=None, \
                                     input_shape=img_shape
                                    )

    drop_layers = ["block4_pool", "block5_pool"]

    input_layer = x = original.input
    for layer in original.layers[1:]:
        if layer.name not in drop_layers:
            x = layer(x)
        elif "block4_conv" in layer.name:
            x = layer(x, dilation_rate=(2,2))
        elif "block5_conv" in layer.name:
            x = layer(x, dilation_rate=(4,4))

    return input_layer, x

def resnet101_backbone(img_shape):
    original = tf.keras.applications.ResNet101(include_top=False, \
                                 weights='imagenet', \
                                 input_tensor=None, \
                                 input_shape=(512,1024,3)
                                )

    # Update the network
    for layer in original.layers:
        if 'conv4' in layer.name or 'conv5' in layer.name:
            # Starting with the 4th block, all strides should be set to (1,1)
            try:
                layer.strides = (1,1)
            except:
                pass
            
            try:
                if layer.kernel_size == (3,3):
                    if 'conv4_block' in layer.name:
                        # In the 4th block, all 3x3 convolutions have dilation_rate = 2
                        layer.dilation_rate = (2,2)
                    elif 'conv5_block' in layer.name:
                        # In the 5th block, all 3x3 convolutions have dilation_rate = 4, except the first one, 
                        # which has dilation_rate = 2
                        layer.dilation_rate = (4,4)

                        # Special case
                        if layer.name == 'conv5_block1_2_conv':
                            layer.dilation_rate = (2,2)
            except:
                pass

    # After the network is changed, it needs to be reloaded
    # Not sure if there is a more clean solution for this
    tmp = np.random.randint(1e10)
    original.save_weights("resnet101_" + str(tmp) + "_temp.h5")
    original = tf.keras.models.model_from_json(original.to_json())
    original.load_weights("resnet101_" + str(tmp) + "_temp.h5")

    # Add the deeplabv3 decoder
    input_layer = x = original.input
    x = original(x)

    return input_layer, x


def deeplabv3(img_shape=(512,1024,3), num_classes=14, backbone = 'vgg16', activation=None):
    assert backbone in ['vgg16', 'resnet101']

    if backbone == 'vgg16':
        input_layer, x = vgg16_backbone(img_shape)

    if backbone == 'resnet101':
        input_layer, x = resnet101_backbone(img_shape)

    x0 = tf.keras.layers.Conv2D(256, 1, use_bias=False)(x)
    x0 = tf.keras.layers.BatchNormalization()(x0)
    x0 = tf.keras.layers.Activation('relu')(x0)

    # ASPP Conv
    dilation=12
    x1 = tf.keras.layers.ZeroPadding2D(padding=dilation)(x)
    x1 = tf.keras.layers.Conv2D(256, 3, dilation_rate=dilation, use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)

    dilation=24
    x2 = tf.keras.layers.ZeroPadding2D(padding=dilation)(x)
    x2 = tf.keras.layers.Conv2D(256, 3, dilation_rate=dilation, use_bias=False)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation('relu')(x2)

    dilation=36
    x3 = tf.keras.layers.ZeroPadding2D(padding=dilation)(x)
    x3 = tf.keras.layers.Conv2D(256, 3, dilation_rate=dilation, use_bias=False)(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Activation('relu')(x3)

    # ASPP Pooling
    size = x.shape[1:3]
    x4 = tf.keras.layers.GlobalAveragePooling2D()(x)
    x4 = tf.keras.layers.Lambda(lambda xx: tf.keras.backend.expand_dims(xx, 1))(x4)
    x4 = tf.keras.layers.Lambda(lambda xx: tf.keras.backend.expand_dims(xx, 1))(x4)
    x4 = tf.keras.layers.Conv2D(256, 1, use_bias=False)(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Activation('relu')(x4)

    x4 = tf.keras.layers.Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                           size,
                                                           method='bilinear', 
                                                           align_corners=False),
                                                           name='pooling_resizing_layer')(x4) 


    x = tf.keras.layers.Concatenate()([x0, x1, x2, x3, x4])

    # Project
    x = tf.keras.layers.Conv2D(256, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(.5)(x)

    # Post Projection
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(256, 3, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_classes, 1)(x)

    # Final resizing
    x = tf.keras.layers.Lambda(lambda x: tf.compat.v1.image.resize(x,
                                                           img_shape[:2],
                                                           method='bilinear', 
                                                           align_corners=False),
                                                           name='final_resizing_layer')(x)

    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)

    model = tf.keras.Model(input_layer, x)

    return model


