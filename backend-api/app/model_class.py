import tensorflow as tf
import tensorflow.keras as K


@K.utils.register_keras_serializable(package='Custom')
def l2norm(x):
    return tf.nn.l2_normalize(x, axis=1)


@K.utils.register_keras_serializable(package='Custom')
def build_feature_extractor(input_shape, backbone, dim=512):
    def self_attention(layer,):
        x = K.layers.Dropout(0.5)(layer)
        x = K.layers.Dense(dim // 2, activation='gelu')(x)
        att_mask = K.layers.Dense(dim, activation='sigmoid')(x)

        x = K.layers.Dropout(0.5)(layer)
        x = K.layers.Dense(dim)(x)

        return x * att_mask
    
    backbone.trainable = False
    
    inputs = K.layers.Input(shape=input_shape)

    # feature extraction
    embeddings = backbone(inputs, training=False)
    
    # vectorize
    x = K.layers.GlobalAveragePooling2D()(embeddings)
    x = self_attention(x)
    outputs = K.layers.Lambda(l2norm)(x)

    feature_extractor = K.Model(inputs, outputs)
    return feature_extractor


@K.utils.register_keras_serializable(package='Custom')
class FaceModel(K.Model):
    def __init__(self, feature_extractor, dim=512, margin=0.1, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor=feature_extractor
        self.margin=margin
        self.dim=dim
        self.triplet_loss_tracker=K.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [
            self.triplet_loss_tracker,
        ]
    
    def compute_loss(self, p, n, q):
        "soft margin semi-hard negatives"

        p_dist = tf.reduce_sum(tf.square(p - q), axis=1) # (batch, dim) -> (batch,)
        n_dist = tf.reduce_sum(tf.square(n - q), axis=1)

        # dp < dn < dp + margin
        # mask = tf.logical_and(p_dist < n_dist, n_dist < (p_dist + self.margin)) # (batch, 1) bool

        diff = p_dist - n_dist # (batch,)
        soft_loss = tf.math.log1p(tf.exp(diff)) # (batch,)
        loss = soft_loss

        # manter apenas as distâncias moderadas
        # loss = tf.boolean_mask(soft_loss, mask) # (batch,)
        return tf.reduce_mean(loss) # em caso de mask ser all False TODO: VERIFICAR MÉDIA COM MASK 

    def train_step(self, data):
        with K.backend.name_scope('train'):                    
            p_episodes, n_episodes, q_episodes = data                                         
            num_episodes = p_episodes.shape[0] # equivalente ao batch_size            
                        
            with tf.GradientTape() as tape:
                # (batch, dim) vai armazenar todas as inferências de cada classe 
                # Criando TensorArrays dinâmicos
                p_batch_pred = tf.TensorArray(tf.float32, size=num_episodes)
                n_batch_pred = tf.TensorArray(tf.float32, size=num_episodes)
                q_batch_pred = tf.TensorArray(tf.float32, size=num_episodes)
                for episode in range(num_episodes): # (# batch iterações)
                    p_inputs, n_inputs, q_inputs = p_episodes[episode], n_episodes[episode], q_episodes[episode]

                    # transforma todas as entradas em um único batch de (3*kshots, H, W, 3)
                    inputs = tf.concat([p_inputs, n_inputs, q_inputs], axis=0) # (P + N + Q)
                    kshots = inputs.shape[0] // 3

                    # extração de todos os vetores
                    embeddings = self.feature_extractor(inputs)
                    
                    # split de cada classe
                    p_embeddings = embeddings[0:kshots]
                    n_embeddings = embeddings[kshots:2*kshots]
                    q_embeddings = embeddings[-kshots:]

                    # médias do K-shot
                    p_vector = tf.reduce_mean(p_embeddings, axis=0)
                    n_vector = tf.reduce_mean(n_embeddings, axis=0)
                    q_vector = tf.reduce_mean(q_embeddings, axis=0)

                    # preencher os tensores dinâmicos
                    p_batch_pred = p_batch_pred.write(episode, p_vector)
                    n_batch_pred = n_batch_pred.write(episode, n_vector)
                    q_batch_pred = q_batch_pred.write(episode, q_vector)
                    
                # após todas as inferências, recuperar o valor final (Batch, dim)
                p_batch_pred = p_batch_pred.stack()
                n_batch_pred = n_batch_pred.stack()
                q_batch_pred = q_batch_pred.stack()
                
                # soft margin semi-hard triplet loss
                loss = self.compute_loss(p_batch_pred, n_batch_pred, q_batch_pred)

            grads = tape.gradient(loss, self.feature_extractor.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.feature_extractor.trainable_weights))

            # atualizar trackers
            self.triplet_loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics}
    
                
    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_extractor": K.utils.serialize_keras_object(self.feature_extractor),
            "margin": self.margin,
            "dim": self.dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        fe_ser = config.pop("feature_extractor")        
        feature_extractor = K.utils.deserialize_keras_object(fe_ser)
        
        margin = config.pop("margin", 0.2)
        dim = config.pop("dim", 512)

        return cls(feature_extractor=feature_extractor, margin=margin, dim=dim, **config)


# ----------------------------
# Build model
# ----------------------------
backbone = K.applications.EfficientNetB0(
    include_top=False,    
    weights='imagenet',
    input_shape=(224, 224, 3)
)
feature_extractor = build_feature_extractor((224, 224, 3), backbone, dim=512)
model = FaceModel(feature_extractor, dim=512, margin=0.2)
model.compile(optimizer=K.optimizers.Adam(1e-3))


# ----------------------------
# Build model
# ----------------------------
# backbone = K.applications.EfficientNetB0(
#     include_top=False,    
#     weights='imagenet',
#     input_shape=(224, 224, 3)
# )
# feature_extractor = build_feature_extractor((224, 224, 3), backbone)
# model = FaceModel(feature_extractor, 0.1)
# model.compile(optimizer=K.optimizers.Adam(1e-3))