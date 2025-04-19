# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization, Lambda, GlobalAveragePooling2D, Add
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf

import time


# Suas classes customizadas:
class L1Dist(Layer):
    def __init__(self, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def call(self, inputs):
        input_embedding, validation_embedding = inputs  # Recebe dois embeddings
        if self.normalize:
            input_embedding = tf.nn.l2_normalize(input_embedding, axis=-1)
            validation_embedding = tf.nn.l2_normalize(validation_embedding, axis=-1)
        return tf.math.abs(input_embedding - validation_embedding)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"normalize": self.normalize})
        return config

class L2Normalize(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super().get_config()



#model = tf.keras.models.load_model('siamese_model_updated.h5', custom_objects={'L1Dist': L1Dist, 'L2Normalize': L2Normalize}, compile=False)  # ou .keras
#model.summary()  # Verificar a estrutura do modelo carregado


def preprocess_image(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.image.resize(image, (256, 256))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def gerar_vetores_por_usuario(model, pasta_raiz, save_path="vetores_por_usuario.npy"):
    """
    Percorre a pasta_raiz (onde cada subpasta representa um usuário) e gera os
    embeddings para todas as imagens de cada usuário.
    Salva um dicionário com chave = nome da subpasta (usuário) e valor = array de vetores.
    """
    vetores_por_usuario = {}
    for usuario in os.listdir(pasta_raiz):
        usuario_path = os.path.join(pasta_raiz, usuario)
        if os.path.isdir(usuario_path):
            vetores = []
            # Lista de arquivos de imagem na pasta do usuário
            for img in os.listdir(usuario_path):
                file_path = os.path.join(usuario_path, img)
                if os.path.isfile(file_path):
                    image = preprocess_image(file_path)
                    image = tf.expand_dims(image, axis=0)  # (1, 256, 256, 1)
                    # Obter o vetor pela camada "Gray_InceptionV3"
                    vetor = model.get_layer("Gray_InceptionV3")(image, training=False).numpy()[0]
                    vetores.append(vetor)
            if vetores:
                # Converte a lista de vetores para um array de dimensão (n, 1024)
                vetores_por_usuario[usuario] = np.stack(vetores, axis=0)
    np.save(save_path, vetores_por_usuario)
    print(f"Vetores agrupados por usuário salvos em: {save_path}")
    return vetores_por_usuario


def carregar_vetores_salvos(file_path="vetores_por_usuario.npy"):
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True).item()
    else:
        return {}


def adicionar_novos_vetores(model, pasta_raiz, save_path="vetores_por_usuario.npy"):
    if os.path.exists(save_path):
        vetores_existentes = np.load(save_path, allow_pickle=True).item()
    else:
        vetores_existentes = {}

    for usuario in os.listdir(pasta_raiz):
        usuario_path = os.path.join(pasta_raiz, usuario)
        if not os.path.isdir(usuario_path):
            continue

        if usuario in vetores_existentes:
            return False

        novos_vetores = []

        for img in os.listdir(usuario_path):
            img_path = os.path.join(usuario_path, img)
            if os.path.isfile(img_path):
                image = preprocess_image(img_path)
                image = tf.expand_dims(image, axis=0)
                vetor = model.get_layer("Gray_InceptionV3")(image, training=False).numpy()[0]
                novos_vetores.append(vetor)

        # Só processa se houver novos vetores
        if novos_vetores:
            novos_array = np.stack(novos_vetores, axis=0)
            vetores_existentes[usuario] = novos_array

    np.save(save_path, vetores_existentes)
    return True





def verificar_imagem_no_sistema(model, image_hand, vetores_por_usuario, limiar=0.8, CORRESPONDENCIA = 4):
    """
    Gera o vetor da imagem de teste e, para cada usuário (grupo de vetores), calcula a saída 
    (usando l1_dist_2 -> dense_4 -> dense_5) para cada vetor do grupo.
    Retorna (True, usuario, acuracia) se algum usuário tiver pelo menos 3 correspondências ou acurácia >= limiar.
    """
    # Pré-processar e obter o vetor da imagem de teste
    image = preprocess_image(image_hand)
    image = tf.expand_dims(image, axis=0)
    vetor_imagem = model.get_layer("Gray_InceptionV3")(image, training=False).numpy()  # (1,1024)
    vetor_imagem_tensor = tf.convert_to_tensor(vetor_imagem)

    melhor_usuario = None
    maior_correspondencia = 0
    melhor_acuracia = 0

    # Itera sobre cada usuário (cada chave do dicionário vetores_por_usuario)
    for usuario, vetores in vetores_por_usuario.items():
        # vetores: array de shape (n, 1024)
        vetores_tensor = tf.convert_to_tensor(vetores)  # shape (n,1024)
        # Passa o vetor da imagem de teste e os vetores do usuário pela camada l1_dist_2
        l1_output = model.get_layer("l1_dist_2")([vetor_imagem_tensor, vetores_tensor])
        # Passa pelas camadas dense para obter o score final para cada vetor
        dense_4_output = model.get_layer("dense_4")(l1_output)
        resultado = model.get_layer("dense_5")(dense_4_output).numpy().flatten()  # array de scores de length n
        
        # Contabiliza quantos dos scores são >= limiar
        count = np.sum(resultado >= limiar)
        total = len(vetores)
        acuracia = count / total

        #print(f"Usuário {usuario}: {count} correspondências de {total} vetores (acurácia: {acuracia:.2f})")

        # Seleciona o usuário com maior número de correspondências
        if count > maior_correspondencia:
            maior_correspondencia = count
            melhor_usuario = usuario
            melhor_acuracia = acuracia

    # Critério de aceitação: por exemplo, se o usuário com maior correspondência tiver pelo menos 4 ou acurácia >= limiar
    if melhor_usuario is not None and (maior_correspondencia >= CORRESPONDENCIA):
        return True, melhor_usuario, melhor_acuracia
    else:
        return False, None, None


#vetores_sistema = carregar_vetores_salvos('vetores_por_usuario.npy')


def verificar_acesso_por_imagem(model, imagem_path, vetores_sistema, limiar=0.9, correspondencia=3):
    """
    Verifica se a imagem corresponde a um usuário do sistema.
    
    Parâmetros:
    - model: modelo de verificação
    - imagem_path: caminho da imagem a ser verificada
    - vetores_sistema: vetores salvos dos usuários
    - limiar: limiar de similaridade
    - correspondencia: número mínimo de correspondências para aceitar o usuário
    
    Retorna:
    - (True, usuario, acuracia) se o acesso for liberado
    - (False, None, None) caso contrário
    """
    inicio = time.time()
    
    acesso, usuario, acuracia = verificar_imagem_no_sistema(
        model, imagem_path, vetores_sistema, 
        limiar=limiar, CORRESPONDENCIA=correspondencia
    )
    
    fim = time.time()
    print(f"Tempo de verificação: {fim - inicio:.2f} segundos")

    if acesso:
        print(f"Acesso liberado para o usuário {usuario}.")
        print(f"Acurácia da correspondência: {acuracia:.2f}")
        return True, usuario, acuracia
    else:
        print("Acesso negado.")
        return False, None, None

