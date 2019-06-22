# Darkteal

<p align="center">
  <img src="logo.png" alt="logo" width="100"/>
<p>

Darkteal is a library to perform some basic scientific computing in (object) pascal. **Still under a heavy development**. It is built purely in pascal language, with OpenBLAS backend for heavy matrix operations.

In the near future, darkteal is aimed to be a foundation to make the development of machine learning algorithm easier. That is why you will notice some neural network-related functions, such as data preprocessing function, a collection of activations, and loss functions.

**Important note:** This library is:
- In a very early development, thus, many missing functionality
- Not for production purpose, since it is not built for performance-critical applications

## Quick Start

### Usage
- Include darkteal's "src" folder into your project search path.
- Put libopenblas shared library (libopenblas.dll) in your project folder. Refer to OpenBLAS [installation guide](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide). Please note that you **must** use libopenblas compiled with LAPACKE enabled.

### Initializing matrices
Darkteal revolves around the usage of TDTMatrix. The TDTMatrix is essentially a wrapper of pascal's 2D dynamic array with OpenBLAS as the backend for some matrix operations. Darkteal provides helper functions to initialize matrices in various ways:
```pascal
program example;

uses
  wincrt,
  SysUtils,
  DTCore; // core include

var
  A, B, C: TDTMatrix;

begin
  A := CreateMatrix(3, 4);     // 3x4 matrix with random values
  B := CreateMatrix(3, 4, 10); // 3x4 matrix filled with 10

  WriteLn('Matrix A: ');
  PrintMatrix(A);
  WriteLn();

  WriteLn('Matrix B: ');
  PrintMatrix(B);
end;
```

It is also possible to create a matrix by loading from a CSV file by using ```TDTMatrixFromCSV``` function. Keep in mind that this CSV loader is not a general purpose CSV loader, i.e., it is designed to load numerical data **only** for the sake of loading speed, at least for now. All numerical values will be converted into floating-point numbers.
```pascalWriteLn(Xtrain.Height);
  ...
  C := TDTMatrixFromCSV('yourfilename.csv');
  PrintMatrix(C);
  ...
```

Followings are the example of matrix operations:
```pascal
  // Transpose
  WriteLn('A transpose:');
  C := A.T;
  PrintMatrix(C);
  writeln();

  // Addition
  WriteLn('A + B:');
  C := Add(A, B); // or C := A + B;
  PrintMatrix(C);
  writeln();

  // Dot product
  WriteLn('<A, B>:');
  C := A.Dot(B.T);
  PrintMatrix(C);
  writeln();

  // Hadamard (element-wise) product
  WriteLn('A .* B:');
  C := Multiply(A, B); // or C := A * B
  PrintMatrix(C);

  ReadLn();  
```
Check out the documentation [here](https://ariaghora.github.io/darkteal/docs/)

### Machine learning (classification) example
There are few machine learning functionalities included. For now, logistic regression and naive bayes classifier are implemented. I do hope more are implemented near future. Following is an example of the usage of naive bayes classifier on fisher iris dataset:
```pascal
uses
  ..., 
  DTCore,               // core unit
  DTMLUtils,            // machine learning utilities
  DTPreprocessing;      // data preprocessing utilities
var
  Dataset, X, y, XTrain, Xtest, yTrain, yTest: TDTMatrix;
  Predictions: TDTMatrix;
  TrainAccuracy, TestAccuracy: double;
begin
  { Load dataset }
  Dataset := TDTMatrixFromCSV('iris_dataset.csv');
  X := Dataset.GetRange(0, 0, Dataset.Height, Dataset.Width - 1); // features
  y := Dataset.GetColumn(Dataset.Width - 1);                      // label

  { Split the dataset into training set and testing set }
  TrainTestSplit(X, y, 0.7, XTrain, Xtest, yTrain, yTest, False);

  { Scale the dataset }
  Scaler := TMinMaxScaler.Create;
  Scaler.Fit(XTrain);
  XTrain := Scaler.Transform(XTrain);
  XTest := Scaler.Transform(Xtest);

  { Initialize and training the classifier }
  NaiveBayes := TClassifierNaiveBayes.Create;
  NaiveBayes.StartTraining(XTrain, yTrain);
  Predictions := NaiveBayes.MakePrediction(Xtest);
  TestAccuracy := AccuracyScore(Predictions, yTest); 

  WriteLn('Test accuracy: ', TestAccuracy);
end.
```
Please check [this link](https://ariaghora.github.io/darkteal/docs/DTMLUtils.html) for a more complete classifier list and some other machine learning-related functionalities. Contributions are welcome.

### Some known issues
- **Some operations are painfully slow:** darkteal is still in a very early development. What you can do for now is making optimization on the compiler side, e.g., using "-O3" if you are using freepascal compiler.
- **Successful matrix operation despite of being under dimension mismatch:** Set the compiler to check assertion.

## License
This project is licensed under the MIT License
