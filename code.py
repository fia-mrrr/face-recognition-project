import cv2
from cvzone.ClassificationModule import Classifier

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
myClassifier = Classifier('converted_keras/keras_model.h5', 'converted_keras/labels.txt')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Распознавание лиц с помощью каскадов Хаара
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Зеркалим кадр
    flip_frame = cv2.flip(frame, 1)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    # Проверяем, что лица были распознаны
    if len(faces) != 0:
        # Получаем предсказания от нашей модели
        predictions, index = myClassifier.getPrediction(flip_frame, draw=False)

        # Отфильтровываем предсказания с меньшими вероятностями
        predictions = [p for p in predictions if p > 0.8]

        # Если есть предсказания с вероятностью больше 80%, выводим их на экран
        if predictions:
            # Выводим процент совпадения на экран
            text = f" {int(max(predictions)*100)}% match"
            cv2.putText(
                flip_frame,
                text,
                (faces[0][0],
                 faces[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )
            cv2.putText(
                flip_frame,
                str(myClassifier.list_labels[index]),
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                2, (0,255,0), 2
            )
        else:
            cv2.putText(
                flip_frame,
                f"Who is it? ",
                (faces[0][0],
                 faces[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2
            )
    # Отображаем кадр на экране
    cv2.imshow("Video", flip_frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
video_capture.release()
cv2.destroyAllWindows()
