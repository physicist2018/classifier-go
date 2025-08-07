(Release)[https://github.com/physicist2018/classifier-go/releases/tag/v1.8.2]
### Описание

Программа выполняет классификацию аэрозоля по трем типам:
- пылевой
- городской
- смоговый

разделение происходит на основании анализа двух параметров:
- Gf - емкость флуоресценции
- $\delta_p$ - аэрозольная деполяризация

Основанием для классификации служит статья (https://doi.org/10.5194/amt-17-3367-2024)

для работы программы нужно указать реперные значения $Gf$, $\delta$ для каждого из рассматриваемых типов аэрозолей (urban, smoke, dust)

### Алгоритм работы

Доля вклада каждого компонента в измеренные значения  $Gf_m$ и  $\delta_m$ решается следующая система линейных уравнений

$$
\begin{cases}
Gf_d \cdot n_d + Gf_u \cdot n_u + Gf_s \cdot n_s = Gf_m \\
\delta'_d \cdot n_d + \delta'_u \cdot n_u + \delta'_s \cdot n_s = \delta_m \\
n_d+n_u+n_s = 1
\end{cases}
$$

При этом $\delta' = \delta/(1+\delta)$ .

Решение этой системы методом наименьших квадратов позволяет разложить влияние каждого из типов аэрозолей в смесь, а также уточнить значения параметров для каждого из классов.


Реперные значения классов аэрозолей мв задаем с некотрой погрешностью, при этом они напрямую отвечают за результирующее решение ($n_d,n_u,n_s$). чтобы получить статистически обоснованное решение и получить уточненные значения параметров классов мы выполняем моделирование, для этого мы создаем N наборов реперных значений ($Gf^i_d,\delta^i_d,Gf^i_u,\delta^i_u,Gf^i_s,\delta^i_s$, $i=1..N$), которые располагаются вблизи изначально заданных классов и имеют нормальное распределение, параметры которого контролируются параметрами запуска программы. Для каждого набора мы решаем систему линейных уравнений. На выходе мы получаем вектор вида:

$$
\alpha_i = \left( \begin{matrix}
n_d & n_u & n_s & F & Gf_d & \delta_d & Gf_u & \delta_u & Gf_s & \delta_s\\
\end{matrix}\right)
$$

Помимо уже известных нам известных нам величин, F - соответствует L2 норме нашей невязки в решении систему уравнений
Далее, мы ранжируем наши векторы решений, в порядке возрастания и берем p% полученных решений с наименьшим значением F и усредняем. В результате мы получим

$$
<\alpha> =\left( \begin{matrix}
n_d & n_u & n_s & F & Gf_d & \delta_d & Gf_u & \delta_u & Gf_s & \delta_s\\
\end{matrix}\right)
$$

вектор, состоящий из средних значений каждого из компонентов

Этот этап повторяется для каждого из измерений ($Gf$, $\delta$)

Полученные результаты оформляются в виде матрицы, которая сохраняется в csv  и pdf

### Параметры командной строки
```bash
./classifier -h

Usage of temp:

  -config string

    path to config file (default "config.yml")

Usage of ./classifier:

  -delta-dust float

    реперное значение аэрозольной деполяризации, соответствующее пыли (центр кластера из статьи) (default 0.26)

  -delta-smoke float

    реперное значение аэрозольной деполяризации, соответствующее смогу (центр кластера из статьи) (default 0.06)

  -delta-urban float

    реперное значение аэрозольной деполяризации, соответствующее городскому аэрозолю (центр кластера из статьи) (default 0.05)

  -gf-dust float
	реперное значение флуоресцентной емкости пыли (центр кластера из статьи)
    fluorescence capacity of dust aerosol (default 3e-05)

  -gf-smoke float
	реперное значение флуоресцентной емкости смогу (центр кластера из статьи)
    fluorescence capacity of soot aerosol (default 0.0004)

  -gf-urban float
	реперное значение флуоресцентной емкости городского аэрозоля (центр кластера из статьи)
    fluorescence capacity of urban aerosol (default 5.5e-05)

  -input-dir string

    путь к папке где лежат файлы (FL_cap.txt, dep.txt) (default "./")

  -num-points int

    размер моделируемого набора данных параметров кластеров  (default 100)
   
 -avg-percent float

    доля профилей для усреднения (default 0.1 от числа, заданноко в num-points )

  -sigma-h int

    полуширина сглаживающего окна по высоте в отсчетах (default 5)

  -sigma-t int

    полуширина сглаживающего окна по времени в отсчетах (default 3)

  -size int

    размер сглаживающего окна в точках (default 7)

  -smooth

    флаг, указывающий необходимость сглаживания (default false)

  -var-coef float

    величина, указывающаа на скольно сильно нужно делать разброс значений при моделировании  набора данных параметров кластеров (default 0.1)
```

Все эти параметры можно задавать явно, либо указать в файле `config.yml`

```yml
gf_urban: 0.55e-4
gf_smoke: 4e-4
gf_dust: 0.3e-4
delta_urban: 0.05
delta_smoke: 0.06
delta_dust: 0.26
variation_coefficient: 0.1
input_dir: "./test"
num_points: 100
do_smooth: true
sigma_h: 5
sigma_t: 3
size: 7
avg_percent: 0.1
```

При запуске программа сначала ищет файл `config.yml`, читает оттуда настройки, потом проверяет, есть ли какие другие параметры, указанные явно. Если есть - применяет их к уже считанным из файла. Если `config.yml` не найден, программа считывает настройки из флагов, если и их нет - использует значения по умолчанию

В результате мы получаем следующие файлы в той же директории, где лежат фходные файлы `FL_cap.txt` и `dep.txt`

- delta_d.csv, delta_u.csv, delta_s.csv - файлы с уточненными значениями delta
- Gf_d.csv, Gf_u.csv, Gf_s .csv- файлы с уточненными значениями емкости флуоресценции
- Eta_d.csv, Eta_u.csv, Eta_s.csv - файлы с рассчитанными вкладами в смесь dust, urban и smoke
- Eta_d.pdf, Eta_u.pdf, Eta_s.pdf - графическое представление данных Eta
- dep.pdf - графическое представление деполяризации
- rand.csv - смоделированные данные для реперных кластеров

### Сглаживание контролируется параметрами
- sigma-h
- sigma-t
- size
- smooth

### Статистика данных контролируется параметрами
- num-points
- var-coef
- avg-percent

### Дополнительные параметры
- variation_coefficient_delta определяет вариацию всех параметров Gf
- variation_coefficient_gf определяет вариацию всех параметров delta
- variation_coefficient удален

### Изменения
- Изменил решатель на BGFS
- Добавил сохранение матрицы погрешностей
- Изменил способ расчета невязки


# К вопросу о моделировании синтетических данных

### Система линейных уравнений

$$
\begin{cases}
Gf_d \cdot n_d + Gf_u \cdot n_u + Gf_s \cdot n_s = Gf_m \\
\delta'_d \cdot n_d + \delta'_u \cdot n_u + \delta'_s \cdot n_s = \delta_m \\
n_d+n_u+n_s = 1
\end{cases}
$$
Подразумевает, что Gf и $\delta'$ различных аэрозольных типов линейно аддитивны, однако это утверждение необходимо проверять:


Чтобы выразить $\delta_t$ через $\delta_u$, $\delta_d$ и $\delta_s$, начнём с исходного выражения:

$$
\delta_t = \frac{\eta_u \beta_s^u + \eta_d \beta_s^d + \eta_s \beta_s^s}{\eta_u \beta_p^u + \eta_d \beta_p^d + \eta_s \beta_p^s}
$$

Выразим $\beta_s^u$, $\beta_s^d$ и $\beta_s^s$ через $\delta_u$, $\delta_d$ и $\delta_s$:

$$
\beta_s^u = \delta_u \beta_p^u, \quad \beta_s^d = \delta_d \beta_p^d, \quad \beta_s^s = \delta_s \beta_p^s
$$

Подставим эти выражения в числитель:

$$
\delta_t = \frac{\eta_u \delta_u \beta_p^u + \eta_d \delta_d \beta_p^d + \eta_s \delta_s \beta_p^s}{\eta_u \beta_p^u + \eta_d \beta_p^d + \eta_s \beta_p^s}
$$

Теперь числитель и знаменатель имеют схожую структуру. Если ввести обозначения для весовых коэффициентов:

$$
w_u = \eta_u \beta_p^u, \quad w_d = \eta_d \beta_p^d, \quad w_s = \eta_s \beta_p^s,
$$

то формула примет вид:

$$
\delta_t = \frac{w_u \delta_u + w_d \delta_d + w_s \delta_s}{w_u + w_d + w_s}
$$

Таким образом, $\delta_t$ является **взвешенным средним** значений $\delta_u$, $\delta_d$ и $\delta_s$ с весами $w_u$, $w_d$ и $w_s$ соответственно.

### Итоговый ответ:
$$
\delta_t = \frac{\eta_u \beta_p^u \delta_u + \eta_d \beta_p^d \delta_d + \eta_s \beta_p^s \delta_s}{\eta_u \beta_p^u + \eta_d \beta_p^d + \eta_s \beta_p^s}
$$


Что имеем по $Gf$

$$
Gf = \frac{\eta_d\beta^f_d+\eta_u\beta^f_u+\eta_s\beta^f_s}{\eta_d\beta^d_{532}+\eta_u\beta^u_{532}+\eta_s\beta^s_{532}}
$$

 Где
$$
Gf_u = \frac{\beta^u_f}{\beta^u_{532}}
$$

$$
Gf_d = \frac{\beta^d_f}{\beta^d_{532}}
$$

$$
Gf_s = \frac{\beta^s_f}{\beta^s_{532}}
$$

Чтобы выразить $G_f$ через $Gf_u = \frac{\beta^u_f}{\beta^u_{532}}$, $Gf_d = \frac{\beta^d_f}{\beta^d_{532}}$ и $Gf_s = \frac{\beta^s_f}{\beta^s_{532}}$, подставим эти соотношения в исходное выражение:

$$
G_f = \frac{\eta_d \beta^f_d + \eta_u \beta^f_u + \eta_s \beta^f_s}{\eta_d \beta^d_{532} + \eta_u \beta^u_{532} + \eta_s \beta^s_{532}}
$$

Выразим $\beta^f_d$, $\beta^f_u$ и $\beta^f_s$ через $Gf_d$, $Gf_u$ и $Gf_s$:

$$
\beta^f_d = Gf_d \cdot \beta^d_{532}, \quad \beta^f_u = Gf_u \cdot \beta^u_{532}, \quad \beta^f_s = Gf_s \cdot \beta^s_{532}
$$


Подставим эти выражения в числитель:

$$
G_f = \frac{\eta_d Gf_d \beta^d_{532} + \eta_u Gf_u \beta^u_{532} + \eta_s Gf_s \beta^s_{532}}{\eta_d \beta^d_{532} + \eta_u \beta^u_{532} + \eta_s \beta^s_{532}}
$$

Теперь числитель и знаменатель имеют схожую структуру. Если ввести весовые коэффициенты:

$$
w_d = \eta_d \beta^d_{532}, \quad w_u = \eta_u \beta^u_{532}, \quad w_s = \eta_s \beta^s_{532},
$$

то формула примет вид:

$$
G_f = \frac{w_d Gf_d + w_u Gf_u + w_s Gf_s}{w_d + w_u + w_s}
$$

### Итоговый ответ:
$$
G_f = \frac{\eta_d \beta^d_{532} Gf_d + \eta_u \beta^u_{532} Gf_u + \eta_s \beta^s_{532} Gf_s}{\eta_d \beta^d_{532} + \eta_u \beta^u_{532} + \eta_s \beta^s_{532}}
$$

Это означает, что $G_f$ является **взвешенным средним** значений $Gf_d$, $Gf_u$ и $Gf_s$, где веса определяются произведением коэффициентов $eta$ и соответствующих $\beta_{532}$.


Но: $w_d=\eta_d\beta^d_{532}=\eta_d\beta^p_d$,  следовательно система уравнений верна, для $\delta$ и $Gf$, зачем вводить $\delta'$ - пока не понятно.

### Исправленная система уравнений

$$
\begin{cases}
Gf_d \cdot n_d + Gf_u \cdot n_u + Gf_s \cdot n_s = Gf_{meas} \\
\delta_d \cdot n_d + \delta_u \cdot n_u + \delta_s \cdot n_s = \delta_{meas} \\
n_d+n_u+n_s = 1
\end{cases}
$$

Исходя их этих соображений, моделировать тестовые данные как линейную комбинацию аэрозолей заданных типов вполне верно
