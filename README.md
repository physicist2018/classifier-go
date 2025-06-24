# classifier
Программа для выполнения классификации данных зондирования на типы воздушных масс

1. Urban
2. Soot
3. Dust

Поддерживается разделение на эти три компонента


Для запуска программы необходимо указать как минимум путь к каталогу, где лежат подготовленные матрицы

Для работы нужно подготовить два файлв:

* Dep.txt
* FL_cap.txt

Оба этих файла должны содержать матрицы одного размера

## Структура файла Dep.txt

строка заголовок, пропускается при чтении и важен только для пользователя
первый столбец - это отсчеты высоты/расстояни

пример файла приведен ниже

значения аэрозольной деполяризации приведены в процентах

```
H	    dep-21-16	dep-21-20	dep-21-2
120	    21.85648	24.28308	25.06555
240	    16.35283	17.57913	17.79655
360	    10.1951     10.35733	10.09052	
480	    8.94443	    8.8002	    8.50619	
```

## Структура файла FL_Cap.txt 
структура этого файла идентична структуре файла `Dep.txt`


## Запуск

```bash
./classifier -input-dir data/
```

слэш в конце  названия каталога обязателен

в этом случае программа будет искать файлы `Dep.txt` и `FL_cap.txt` в каталоге `data/`

По окончании расчетов в этой папке будут созданы файлы `Eta_d.csv`, `Eta_u.csv`  и `Eta_s.csv` с матрицами 
содержащими относительный вклад каждого из компонентов

* Eta_d - Dust
* Eta_u - Urban
* Eta_s - Soot

Там же будут созданы файлы `Eta_d.pdf`, `Eta_u.pdf`  и `Eta_s.pdf` с развертками относительного ыклада по времени

## Использование

```
Usage of ./classifier:
  -delta-dust float
    	aerosol depolarization for dust aerosol (default 0.26)
  -delta-soot float
    	aerosol depolarization for soot aerosol (default 0.06)
  -delta-urban float
    	aerosol depolarization for urban aerosol (default 0.05)
  -gf-dust float
    	fluorescence capacity of dust aerosol (default 3e-05)
  -gf-soot float
    	fluorescence capacity of soot aerosol (default 0.0004)
  -gf-urban float
    	fluorescence capacity of urban aerosol (default 5.5e-05)
  -input-dir string
    	input directory (default "./")
  -num-points int
    	number of data points to simulate (default 100)
  -var-coef float
    	variation coefficient for aerosol parameters (default 0.1)
  -sigma-h int
    	spatial smoothing size in bins (default 5)
  -sigma-t int
    	temporal smoothing size in bins (default 3)
  -size int
    	kernel size in bins (default 7)
  -smooth
    	apply smoothing to the data

```

## ВАЖНО
Входные данные не толжны соделжать нули, nan, проапуски и отрицательные значения, если таковые есть
или удаляе всю строку, или как-то фиксим значения

## Кастомизация
Программа позваоляет выполнить настройку типовых значений Dust, Urban и Soot для вашего региона посредством аргументов командной строки (см выше)

