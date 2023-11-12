# ВИЖН – интеллектуальный анализ видео

## Запуск моделей
В файлах back_x3d.py и back_resnet.py находится код для запуска моделей на одном видео. Выдается топ-5 предсказаний с вероятностями по каждому из классов. Чтобы запустить скрипт, нужно указать в переменной DATA_DIR путь до видео, по которому делается предсказание, а в WEIGHTS – путь к весам модели.

Веса моделей можно скачать по ссылкам:
- X3D-M – https://www.dropbox.com/scl/fi/f5337bhh7rsvwchixwg02/x3d_m_ep4_0.8162.pt?rlkey=34fqf8grnlrw8v7ct8zs1l18l&dl=0
- 3D-ResNet50 – https://www.dropbox.com/scl/fi/faasaev125gb3c5ntl51y/res3d_weights?rlkey=g8fmrj2v4io5ysol23g6ew2in&dl=0
