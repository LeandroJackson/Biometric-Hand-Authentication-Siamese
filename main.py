import cv2
import urllib.request
import numpy as np
from phscd import phscd
import model
import os
import tensorflow as tf
import time
from ultralytics import YOLO

# Defina o IP da ESP32-CAM
esp32_ip = "192.168.68.126"
model_siamese = tf.keras.models.load_model('siamese_model_updated.h5', custom_objects={'L1Dist': model.L1Dist, 'L2Normalize': model.L2Normalize}, compile=False)  # ou .keras
path_caracteristicas = 'vetores_users.npy'
vetores_sistema = model.carregar_vetores_salvos(path_caracteristicas)
model_seg = YOLO("phscd/model/yolo_hand_phscd.pt")
# URL da captura de imagem

url_esp = f"http://{esp32_ip}/capture"
url_cell = "http://10.0.0.241:8080/video"

cap = cv2.VideoCapture(url_cell)

camera = 1 #0 - camera da ESP
           #1 - camera do celular, app Ip Webcam

def input_save(img, save_path='valid/nome_usuario/'):
    seg, coordinates, mascara = phscd.maskHand(img, model_seg)

    if seg is None:
        return None, False, False, False

    gray = cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)

    mask = gray <= 5
    clahe_img[mask] = 0

    os.makedirs(save_path, exist_ok=True)

    i = 1
    while True:
        filename = f"hand{i}.jpg"
        full_path = os.path.join(save_path, filename)
        if not os.path.exists(full_path):
            break
        i += 1

    # Salvar como JPG com qualidade alta
    cv2.imwrite(full_path, clahe_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    print(f"Imagem salva em: {full_path}")
    return True, coordinates, mascara, True


def input_test(img):
    seg, coordinates, mascara = phscd.maskHand(img, model_seg)

    if seg is None:
        return None, False, False, False

    gray = cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)

    mask = gray <= 5
    clahe_img[mask] = 0

    return clahe_img, coordinates, mascara, True


def capturar_foto(tipo=0):
    if tipo == 0:
        try:
            resp = urllib.request.urlopen(url_esp)
            img_array = np.array(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)  
            return img
        except Exception as e:
            print("Erro ao capturar imagem:", e)
            return None
    else:
        temp_cap = cv2.VideoCapture(url_cell)
        #time.sleep(0.5)  # pequena pausa para garantir que o stream esteja pronto
        ret, frame = temp_cap.read()
        temp_cap.release()
        if not ret:
            print("Erro ao ler frame da câmera do celular.")
            return None
        return frame
    
def view():
    img1 = capturar_foto(camera)
    _, mask = phscd.detect_hand_info(img1, model_seg)

    img1 = cv2.resize(img1, (800, 600))

    if _ == None:
        #cv2.imshow(window_name, img1)  # Exibe a imagem capturada
        return img1
    else:
        mask = cv2.resize(mask, (800, 600))
        img1 = phscd.desenhar_triangulos_biometria(img1, mask)
    
        return img1
        #cv2.imshow(window_name, img1)  # Exibe a imagem capturada 


def main():
    global vetores_sistema
    global camera
    print("Pressione 'f' para tirar uma foto ou 'q' para sair.")

    # Cria uma janela preta
    window_name = "ESP32-CAM Captura"
    black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imshow(window_name, black_screen)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if camera == 1:
            img1 = view()
            cv2.imshow(window_name, img1)  

        if key == ord('s'):
            img = capturar_foto(camera)
            hand, coordinatos, mascara, passou = input_save(img)

            img = cv2.resize(img, (800, 600))


            if passou == True:
                mascara = cv2.resize(mascara, (800, 600))
                img = phscd.desenhar_triangulos_biometria(img, mascara)

                print("MAO SALVA")
                cv2.rectangle(img, (50, 50), (400, 150), (255, 0, 0), -1)
                cv2.putText(img, f"Usuario: ", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                cv2.putText(img, "SALVO", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                cv2.imshow(window_name, img)  
                time.sleep(2) 


            else:
                cv2.rectangle(img, (50, 50), (400, 150), (0, 0, 255), -1)
                cv2.putText(img, "MAO NAO DETECTADA", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                print("MAO NÃO ENCONTRADA, POR FAVOR, TENTE NOVAMENTE")

                cv2.imshow(window_name, img)  
        
        if key == ord('c'):
            salvo = model.adicionar_novos_vetores(model_siamese, 'valid', save_path=path_caracteristicas)
            if salvo == True:
                vetores_sistema = model.carregar_vetores_salvos(path_caracteristicas)
                print("USUARIO ADICIONADO AO BANCO DE DADOS")
            else:
                print("OCORREU UM ERRO AO SALVAR OS NOVOS USUARIOS")




        if key == ord('f'):
            img = capturar_foto(camera)
            hand, coordinatos, mascara, passou = input_test(img)

            img = cv2.resize(img, (800, 600))
            
            if passou == False:
                print("MAO NÃO ENCONTRADA, POR FAVOR, TENTE NOVAMENTE")
                cv2.rectangle(img, (50, 50), (400, 150), (0, 255, 0), -1)
                cv2.putText(img, "MAO NAO DETECTADA", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                cv2.imshow(window_name, img) 
                time.sleep(2) 

            else:
                mascara = cv2.resize(mascara, (800, 600))
                img = phscd.desenhar_triangulos_biometria(img, mascara)

                cv2.imwrite('TESTE/hand.jpg', hand, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                acesso, usuario, acuracia = model.verificar_imagem_no_sistema(
                    model_siamese, 'TESTE/hand.jpg', vetores_sistema, limiar=0.95, CORRESPONDENCIA=4)


                if acesso:
                    print(f"Acesso liberado para o usuário {usuario}.")
                    print(f"Acurácia da correspondência: {acuracia:.2f}")
                    
                    cv2.rectangle(img, (50, 50), (400, 150), (0, 255, 0), -1)
                    cv2.putText(img, f"{usuario}", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                    cv2.putText(img, "LIBERADO", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                    time.sleep(2) 
                else:
                    print("Acesso negado.")

                    cv2.rectangle(img, (50, 50), (300, 130), (0, 0, 255), -1)
                    cv2.putText(img, "NEGADO", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                cv2.imshow(window_name, img) 
                time.sleep(2) 


        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
