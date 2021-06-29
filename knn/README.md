## KNN from scratch

### dataset used: IRIS : Iris.csv

How do I Run this?
```bash
# make sure you are in the root of this project
python ./knn/knn.py
```

Sample Output:
```bash
[2021-06-28 23:52:42,571] [__main__:79] | [INFO]: --- Begin Loading DataSet --- 
[2021-06-28 23:52:42,572] [__main__:83] | [INFO]: Training-Set SIZE: 94
[2021-06-28 23:52:42,572] [__main__:84] | [INFO]: Test-Set SIZE: 55
[2021-06-28 23:52:42,572] [__main__:85] | [INFO]: --- Loading DataSet Completed.--- 
[2021-06-28 23:52:42,572] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,572] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,573] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,573] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,573] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,573] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,573] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,574] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,574] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,574] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,574] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,574] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,575] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,575] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,575] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,575] [__main__:94] | [INFO]: predicted='Iris-setosa', actual='Iris-setosa'
[2021-06-28 23:52:42,575] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,576] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,576] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,576] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,576] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,576] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,576] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,577] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,577] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,577] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,577] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,577] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,578] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,578] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,578] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,578] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,578] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,579] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,579] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,579] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,579] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,579] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,580] [__main__:94] | [INFO]: predicted='Iris-versicolor', actual='Iris-versicolor'
[2021-06-28 23:52:42,580] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,580] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,580] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,580] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,581] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,581] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,581] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,581] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,581] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,582] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,582] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,582] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,582] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,582] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,583] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,583] [__main__:94] | [INFO]: predicted='Iris-virginica', actual='Iris-virginica'
[2021-06-28 23:52:42,583] [__main__:96] | [INFO]: Accuracy: 100.0%
```
