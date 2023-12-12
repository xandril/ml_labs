## Лабораторные работы по курсу "Нейронные сети и машинное обучение" НГУ ФИТ, 2 курс магистратуры 2022-2024

# Задача 1(lab1_backprop)

### Условия:

На основании функции по варианту необходимо:

1. Построить nn.Module, в котором вы определите `forward`и `my_forward_backward`
2. `forward` должен повторять функцию, выданную вам по варианту, `my_forward_backward` описывает проход по
   вычислителному графу, а также вычисление градиентов по этому графу с помощью backprop. Градиенты должны быть
   рассчитаны для параметров $w0, w1$, тензоры $x1, x2, x3$ считаются входными данными сети
3. Необходимо удалять неиспользуемые тензоры, как это делает Pytorch.
4. Если какие-то узлы не нужны для вычисления результата, то вы не должны их вычислять в процессе backprop.
5. Необходимо построить визуализацию вычислительного графа.

### описание пакета lab1_backprop

1. data/comp_graph_graph.png - вычислительный граф, сгенерированный  https://netron.app/
2. data/graph.onnx - модель, использованная для создания вычислительного графа
3. tests/test_graph_model - тест со сравнением результатов торча и модели Graph
4. graph.py - содержит класс Graph - класс модели для вычисления функции и её градиентов по варианту
5. export_to_onnx - скрипт для генерации модели в onnx
6. task_notebook_with_example.ipynb - исходный жупитер ноутбук с заданием

