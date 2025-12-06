import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.gray_image = None
        self.blurred_image = None
        self.gradient_magnitude = None
        self.gradient_angle = None
        self.suppressed_image = None
        self.final_image = None
    
    def task1(self, image_path, blur_kernel_size=5, sigma=1.0):
        """Задание 1: Чтение, перевод в ЧБ, размытие Гаусса и вывод"""
        # Чтение изображения
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Ошибка: Не удалось загрузить изображение по пути {image_path}")
            return
        
        # Перевод в черно-белый
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Применение размытия Гаусса
        self.blurred_image = cv2.GaussianBlur(self.gray_image, 
                                            (blur_kernel_size, blur_kernel_size), 
                                            sigma)
        
        # Вывод результатов
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.title('Исходное изображение')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(self.gray_image, cmap='gray')
        plt.title('Черно-белое изображение')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(self.blurred_image, cmap='gray')
        plt.title(f'Размытие Гаусса (ядро={blur_kernel_size}, σ={sigma})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return self.blurred_image
    
    def task2(self):
        """Задание 2: Вычисление матриц градиентов"""
        if self.blurred_image is None:
            print("Сначала выполните задание 1!")
            return
        
        # Вычисление градиентов по осям X и Y с помощью оператора Собеля
        gradient_x = cv2.Sobel(self.blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(self.blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Матрица длин градиентов (magnitude)
        self.gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Матрица углов градиентов (angle) в радианах
        self.gradient_angle = np.arctan2(gradient_y, gradient_x)
        
        # Преобразование углов в градусы для лучшей интерпретации
        gradient_angle_degrees = np.degrees(self.gradient_angle) % 180
        
        # Вывод результатов
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.blurred_image, cmap='gray')
        plt.title('Размытое изображение')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(self.gradient_magnitude, cmap='hot')
        plt.title('Матрица длин градиентов')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(gradient_angle_degrees, cmap='hsv')
        plt.title('Матрица углов градиентов (градусы)')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Размер матрицы длин: {self.gradient_magnitude.shape}")
        print(f"Максимальная длина градиента: {np.max(self.gradient_magnitude):.2f}")
        print(f"Минимальная длина градиента: {np.min(self.gradient_magnitude):.2f}")
        print(f"Средняя длина градиента: {np.mean(self.gradient_magnitude):.2f}")
        
        return self.gradient_magnitude, self.gradient_angle
    
    def non_maximum_suppression(self, magnitude, angle):
        """Подавление немаксимумов"""
        M, N = magnitude.shape
        suppressed = np.zeros((M, N), dtype=np.float64)
        
        # Преобразование углов в направления (0°, 45°, 90°, 135°)
        angle_degrees = np.degrees(angle) % 180
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    # Определение направления градиента
                    if (0 <= angle_degrees[i, j] < 22.5) or (157.5 <= angle_degrees[i, j] <= 180):
                        neighbors = [magnitude[i, j-1], magnitude[i, j+1]]  # горизонталь
                    elif 22.5 <= angle_degrees[i, j] < 67.5:
                        neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]  # диагональ 45°
                    elif 67.5 <= angle_degrees[i, j] < 112.5:
                        neighbors = [magnitude[i-1, j], magnitude[i+1, j]]  # вертикаль
                    else:  # 112.5 <= angle_degrees[i, j] < 157.5
                        neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]  # диагональ 135°
                    
                    # Подавление немаксимумов
                    if magnitude[i, j] >= max(neighbors):
                        suppressed[i, j] = magnitude[i, j]
                    else:
                        suppressed[i, j] = 0
                except IndexError:
                    continue
        
        return suppressed
    
    def task3(self):
        """Задание 3: Подавление немаксимумов"""
        if self.gradient_magnitude is None or self.gradient_angle is None:
            print("Сначала выполните задание 2!")
            return
        
        # Применение подавления немаксимумов
        self.suppressed_image = self.non_maximum_suppression(
            self.gradient_magnitude, self.gradient_angle
        )
        
        # Нормализация для отображения
        suppressed_normalized = cv2.normalize(self.suppressed_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Вывод результатов
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.gradient_magnitude, cmap='gray')
        plt.title('Матрица длин градиентов')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(self.suppressed_image, cmap='gray')
        plt.title('После подавления немаксимумов')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Бинаризация для лучшей визуализации краев
        edges_visual = (self.suppressed_image > np.mean(self.suppressed_image)).astype(np.uint8) * 255
        plt.imshow(edges_visual, cmap='gray')
        plt.title('Визуализация краев')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Выводы по подавлению немаксимумов:")
        print("1. Уменьшилась толщина обнаруженных границ")
        print("2. Сохранились только локальные максимумы в направлении градиента")
        print("3. Улучшилась точность определения положения границ")
        print(f"Количество ненулевых пикселей до подавления: {np.count_nonzero(self.gradient_magnitude)}")
        print(f"Количество ненулевых пикселей после подавления: {np.count_nonzero(self.suppressed_image)}")
        
        return self.suppressed_image
    
    def double_threshold(self, image, low_ratio=0.1, high_ratio=0.3):
        """Двойная пороговая фильтрация"""
        high_threshold = np.max(image) * high_ratio
        low_threshold = high_threshold * low_ratio
        
        strong_edges = (image >= high_threshold)
        weak_edges = (image >= low_threshold) & (image < high_threshold)
        
        result = np.zeros_like(image)
        result[strong_edges] = 255  # Сильные границы
        
        # Связывание слабых границ с сильными
        M, N = image.shape
        for i in range(1, M-1):
            for j in range(1, N-1):
                if weak_edges[i, j]:
                    # Если слабая граница соединена с сильной
                    if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                        result[i, j] = 255
        
        return result
    
    def task4(self, low_threshold_ratio=0.1, high_threshold_ratio=0.3):
        """Задание 4: Двойная пороговая фильтрация"""
        if self.suppressed_image is None:
            print("Сначала выполните задание 3!")
            return
        
        # Применение двойной пороговой фильтрации
        self.final_image = self.double_threshold(
            self.suppressed_image, low_threshold_ratio, high_threshold_ratio
        )
        
        # Вывод результатов
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.suppressed_image, cmap='gray')
        plt.title('После подавления немаксимумов')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(self.final_image, cmap='gray')
        plt.title('Двойная пороговая фильтрация')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Сравнение с оригиналом
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.imshow(self.final_image, cmap='jet', alpha=0.5)
        plt.title('Наложение границ на оригинал')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Параметры фильтрации:")
        print(f"Нижний порог: {np.max(self.suppressed_image) * low_threshold_ratio:.2f}")
        print(f"Верхний порог: {np.max(self.suppressed_image) * high_threshold_ratio:.2f}")
        print(f"Обнаружено граничных пикселей: {np.count_nonzero(self.final_image)}")
        
        return self.final_image
    
    def task5_experiment(self, image_path):
        """Задание 5: Эксперименты с параметрами"""
        print("=== ЭКСПЕРИМЕНТЫ С ПАРАМЕТРАМИ ===")
        
        kernel_sizes = [3, 5, 7]
        sigmas = [0.5, 1.0, 2.0]
        
        plt.figure(figsize=(15, 10))
        
        for i, kernel_size in enumerate(kernel_sizes):
            for j, sigma in enumerate(sigmas):
                self.task1(image_path, kernel_size, sigma)
                self.task2()
                self.task3()
                final = self.task4(0.1, 0.3)

                idx = i * len(sigmas) + j + 1
                plt.subplot(len(kernel_sizes), len(sigmas), idx)
                plt.imshow(final, cmap='gray')
                plt.title(f'Ядро: {kernel_size}, σ: {sigma}')
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Тестирование различных порогов
        print("\n=== ТЕСТИРОВАНИЕ ПОРОГОВ ===")
        threshold_combinations = [
            (0.05, 0.2),  # низкие пороги
            (0.1, 0.3),   # средние пороги
            (0.2, 0.4)    # высокие пороги
        ]
        
        plt.figure(figsize=(15, 5))
        
        for i, (low_ratio, high_ratio) in enumerate(threshold_combinations):
            self.task1(image_path, 5, 1.0)
            self.task2()
            self.task3()
            final = self.task4(low_ratio, high_ratio)
            
            plt.subplot(1, 3, i+1)
            plt.imshow(final, cmap='gray')
            plt.title(f'Пороги: {low_ratio}/{high_ratio}\nПикселей: {np.count_nonzero(final)}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Пример использования
def main():
    processor = ImageProcessor()
    
    image_path = "sample_image.jpg"  
    print("=== ЗАДАНИЕ 1 ===")
    processor.task1(image_path, blur_kernel_size=5, sigma=1.0)
    
    print("\n=== ЗАДАНИЕ 2 ===")
    processor.task2()
    
    print("\n=== ЗАДАНИЕ 3 ===")
    processor.task3()
    
    print("\n=== ЗАДАНИЕ 4 ===")
    processor.task4(low_threshold_ratio=0.1, high_threshold_ratio=0.3)
    
    print("\n=== ЗАДАНИЕ 5 ===")
    processor.task5_experiment(image_path)

if __name__ == "__main__":
    main()
