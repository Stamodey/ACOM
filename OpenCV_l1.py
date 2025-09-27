"""
лаба 1
Итоговый список заданий:
Задание 1 Установить библиотеку OpenCV.
Задание 2 Вывести на экран изображение. Протестировать три возможных расширения, три различных флага для создания окна и три различных флага для чтения изображения.
Задание 3 Отобразить видео в окне. Рассмотреть методы класса VideoCapture и попробовать отображать видео в разных форматах, в частности размеры и цветовая гамма.
Задание 4 Записать видео из файла в другой файл.
Задание 5 Прочитать изображение, перевести его в формат HSV. Вывести на экран два окна, в одном изображение в формате HSV, в другом – исходное изображение.
Задание 6 (самостоятельно) Прочитать изображение с камеры. Вывести в центре на экране Красный крест в формате, как на изображении. Указать команды, которые позволяют это сделать.
Задание 7 (самостоятельно) Отобразить информацию с вебкамеры, записать видео в файл, продемонстрировать видео.
Задание 8 (самостоятельно) Залить крест одним из 3 цветов – красный, зеленый, синий по следующему правилу: НА ОСНОВАНИИ ФОРМАТА RGB определить, центральный пиксель ближе к какому из цветов красный, зеленый, синий и таким цветом заполнить крест.
Задание 9 (самостоятельно). Подключите телефон, подключитесь к его камере, выведете на экран видео с камеры. Продемонстрировать процесс на ноутбуке преподавателя и своем телефоне.
"""

import cv2
import sys
import os
import time
import argparse

def wait_key_or_quit(delay=0):
    k = cv2.waitKey(delay) & 0xFF
    return k == ord('q')

def ensure_file(path):
    if not os.path.exists(path):
        print(f"[ERROR] Файл не найден: {path}")
        sys.exit(1)

def scale_image(img, max_width=1920, max_height=1080):
    h, w = img.shape[:2]
    scale = min(max_width/w, max_height/h, 1)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

# -------------------------
#  Задание 2 Вывести на экран изображение.
# -------------------------

def task2_show_image_tests(sample_base='sample_image'):
    exts = ['.jpg', '.png', '.webp']
    window_flags = [
        ('WINDOW_NORMAL', cv2.WINDOW_NORMAL),
        ('WINDOW_AUTOSIZE', cv2.WINDOW_AUTOSIZE),
        ('WINDOW_GUI_EXPANDED', cv2.WINDOW_GUI_EXPANDED)
    ]
    read_flags = [
        ('IMREAD_COLOR', cv2.IMREAD_COLOR),
        ('IMREAD_GRAYSCALE', cv2.IMREAD_GRAYSCALE),
        ('IMREAD_REDUCED_COLOR_2', cv2.IMREAD_REDUCED_COLOR_2)
    ]

    for ext in exts:
        path = sample_base + ext
        if not os.path.exists(path):
            print(f"[WARNING] No file {path}, skipping.")
            continue
        print(f"\n Format: {ext}")
        
        for rname, rflag in read_flags:
            img = cv2.imread(path, rflag)
            
            if img is None:
                continue
            
            img = scale_image(img)

            for wname, wflag in window_flags:
                win = f"{os.path.basename(path)} | {rname} | {wname}"
                cv2.namedWindow(win, wflag)

                if wflag == cv2.WINDOW_NORMAL:
                    h, w = img.shape[:2]
                    cv2.resizeWindow(win, w, h)
                
                cv2.imshow(win, img)
                print(f"Displaying: {win} — press any key (q to quit)")
                
                if wait_key_or_quit(0):
                    cv2.destroyAllWindows()
                    return
                
                cv2.destroyWindow(win)
    cv2.destroyAllWindows()
    print("Task 2 is completed")
                

# -------------------------
#  Задание 3 Отобразить видео в окне
# -------------------------

def task3_video_display_resized(video_path='sample_video.mp4',
                                show_gray=True,
                                sizes=((640,360), (320,180), (480,270))):
    
    ensure_file(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return print("[ERROR] Failed to open the video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_size, gray_size, hsv_size = sizes

        frame_orig = cv2.resize(frame, orig_size)
        cv2.imshow("Video - Original", frame_orig)

        if show_gray:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, gray_size)
            cv2.imshow("Video - Gray", gray_resized)

        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_resized = cv2.resize(hsv, hsv_size)
        cv2.imshow("Video - HSV", hsv_resized)

        if wait_key_or_quit(30):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Task 3 is completed")


# -------------------------
#  Задание 4 Записать видео из файла в другой файл.
# -------------------------

def task4_copy_video(in_path='sample_video.mp4', 
                     out_path='out_copy.avi', 
                     codec='XVID'):
    
    ensure_file(in_path)
    cap = cv2.VideoCapture(in_path)

    if not cap.isOpened(): 
        return print("[ERROR] cannot open input video")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, (w,h))
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"[DONE] Copy saved in {out_path}")
    print("Task 4 is completed")

# -------------------------
#  Задание 5 Прочитать изображение, перевести его в формат HSV.
# -------------------------
def task5_show_hsv(image_path='sample_image.jpg'):
    ensure_file(image_path)
    img = scale_image(cv2.imread(image_path, cv2.IMREAD_COLOR))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Original", img)
    cv2.imshow("HSV", hsv)
    
    while True:
        if wait_key_or_quit(0): 
            break
    
    cv2.destroyAllWindows()
    print("Task 5 is completed")

# -------------------------
#  Задание 6 Вывести в центре на экране Красный крест
# -------------------------

def task6_camera_cross(cap_index=0):
    cap = cv2.VideoCapture(cap_index)
    if not cap.isOpened():
        return print("[ERROR] Unable to open the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        length = min(w, h) // 6
        thickness = max(3, min(w, h) // 60)
        color = (0, 0, 255)

        cv2.line(frame, (cx - length, cy), (cx + length, cy), color, thickness)
        cv2.line(frame, (cx, cy - length), (cx, cy + length), color, thickness)

        cv2.imshow("Camera with cross", frame)

        if wait_key_or_quit(1):
            print("[INFO] Exit on 'q'")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Task 6 is completed")

# -------------------------
#  Задание 7 Отобразить информацию с вебкамеры, записать видео в файл, продемонстрировать видео.
# -------------------------

def task7_record_camera(out_cam_path='camera_record.avi', cap_index=0, codec='XVID', record_seconds=5):
    cap = cv2.VideoCapture(cap_index)
    
    if not cap.isOpened(): 
        return print("[ERROR] Unable to open the camera.")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    out = cv2.VideoWriter(out_cam_path, cv2.VideoWriter_fourcc(*codec), fps, (w,h))
    start = time.time()

    while True:
        ret, frame = cap.read()
        
        if not ret: 
            break
        
        out.write(frame)
        cv2.imshow("Recording", frame)
        
        if wait_key_or_quit(1) or time.time()- start > record_seconds: 
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[DONE] File recorded:", out_cam_path)
    print("Task 7 is completed")

# -------------------------
#  Задание 8 Залить крест одним из 3 цветов – красный, зеленый, синий по следующему правилу: НА ОСНОВАНИИ ФОРМАТА RGB определить, центральный пиксель ближе к какому из цветов красный, зеленый, синий и таким цветом заполнить крест.
# -------------------------

def task8_fill_cross_by_center(image_path='sample_image.jpg'):
    ensure_file(image_path)
    img = scale_image(cv2.imread(image_path, cv2.IMREAD_COLOR))

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    b, g, r = img[cy, cx]
    if r >= g and r >= b:
        color = (0, 0, 255)  
    elif g >= r and g >= b:
        color = (0, 255, 0)  
    else:
        color = (255, 0, 0) 

    length = min(w, h) // 6
    thickness = max(10, min(w, h) // 20)

    
    img[cy-thickness//2:cy+thickness//2, cx-length:cx+length] = color
    img[cy-length:cy+length, cx-thickness//2:cx+thickness//2] = color

    cv2.imshow("Cross by center color", img)
    
    while True:
        if wait_key_or_quit(0):
            break

    cv2.destroyAllWindows()
    print("Task 8 is completed")

# -------------------------
#  Задание 9  Подключите телефон, подключитесь к его камере, выведете на экран видео с камеры. Продемонстрировать процесс на ноутбуке преподавателя и своем телефоне.
# -------------------------

def task9_phone_ip_camera(stream_url="http://192.168.1.67:8080/video", out_path=None):
  
    if not stream_url:
        print("[INFO] Укажите stream_url для подключения")
        return

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("[ERROR] Не удалось подключиться к потоку:", stream_url)
        return

    print("[INFO] Подключено к потоку:", stream_url)
    print("[INFO] Для выхода нажмите 'q'")

    # Создаем VideoWriter, если указан out_path
    out = None
    if out_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
        print(f"[INFO] Запись видео в файл: {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Не удалось прочитать кадр")
            break

        cv2.imshow("Phone camera", frame)
        if out: 
            out.write(frame)

        if wait_key_or_quit(1):
            print("[INFO] Выход по 'q'")
            break

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    print("Задание 9 завершено.")

def main():
    parser = argparse.ArgumentParser(description="Лабораторная OpenCV - задания 2..9")
    parser.add_argument('--task', type=int, default=None, help="Номер задания (2..9)")
    parser.add_argument('--image', type=str, default='sample_image.jpg', help="Путь к изображению")
    parser.add_argument('--video', type=str, default='sample_video.mp4', help="Путь к видео")
    parser.add_argument('--out', type=str, default='out.avi', help="Выходной файл")
    parser.add_argument('--stream_url', type=str, default=None, help="URL потоковой веб-камеры")
    parser.add_argument('--cap', type=int, default=0, help="Индекс камеры")
    args = parser.parse_args()

    if args.task is None:
        print("=== Лабораторная OpenCV ===")
    try:
        args.task = int(input("Введите номер задания (2..9): "))
    except ValueError:
        print("[ERROR] Неверный ввод")
        return

    if args.task == 2:
        task2_show_image_tests(os.path.splitext(args.image)[0])
    elif args.task == 3:
        task3_video_display_resized(video_path=args.video)
    elif args.task == 4:
        task4_copy_video(in_path=args.video, out_path=args.out)
    elif args.task == 5:
        task5_show_hsv(image_path=args.image)
    elif args.task == 6:
        task6_camera_cross(cap_index=args.cap)
    elif args.task == 7:
        task7_record_camera(out_path=args.out, cap_index=args.cap, record_seconds=8)
    elif args.task == 8:
        task8_fill_cross_by_center(image_path=args.image)
    elif args.task == 9:
        task9_phone_ip_camera(stream_url=args.stream_url, out_path=args.out)
    else:
        print("[ERROR] Неподдерживаемое задание:", args.task)
if __name__ == '__main__':
    main()
