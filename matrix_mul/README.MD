Описание алгоритма:
Гриды с блоками и нитями я пытаюсь представить в виде матрицы. Поэтому размер блока по x и y будет равен 16 для того, чтобы в одном блоке находилось 256 нитей.
Грид сайз рассчитывается по стандартной формуле.
В функции ядра для каждого потока вычисляется его строчка и столбец. Каждый поток высчитывает элемент в матрице результатов по индексам самого потока. Один поток выполняет суммирование произведений соответствующих элементов матрицы для нахождения одного элемента в результирующей матрице.
Эксперименты:

![image](https://user-images.githubusercontent.com/49097229/204384724-87ab2074-c531-48ea-b6f1-f02d2ffc6ee9.png)
![image](https://user-images.githubusercontent.com/49097229/204384789-c8261740-eca1-4649-b34e-cf6a77d713a1.png)
По таблице и графику отчётливо видно, что параллельный алгоритм сразуже даёт огромное ускорение по сравнению с последовательным. Таким образом, становится понятно, что матричное произведение нужно обязательно паралелить.
