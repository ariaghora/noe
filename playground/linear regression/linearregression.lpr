{
  This program demonstrates the usage of darkteal library to perform linear
  regression based on gradient descent algorithm for optimization.
}

program linearregression;

uses
  DTCore;

var
  dataset, X, y, yHat: TDTMatrix;
  Err, theta, intercept, dJTheta: TDTMatrix;
  Pairs: TDTMatrix;
  i, m: integer;
  learningRate, J: double;

begin
  // load toy dataset
  dataset := TDTMatrixFromCSV('../../datasets/regression_1_var.csv');

  X := GetColumn(dataset, 0); // independent variable
  y := GetColumn(dataset, 1); // target

  // get number of samples
  m := X.Height;

  // Add intercept term to X
  intercept := CreateMatrix(m, 1, 1);

  X := InsertColumnsAt(X, intercept, 0);

  // model parameter, 2x1 matrix initialized with 0.5
  theta := CreateMatrix(2, 1, 0.5);

  WriteLn('Initial theta:');
  PrintMatrix(theta);

  // set the learning rate
  learningRate := 0.003;

  // start gradient descent
  for i := 1 to 100 do
  begin
    yHat := X.Dot(theta); // make prediction

    // calculate mean squared error loss
    J := Mean(Power(yHat - y, 2));

    if i mod 10 = 0 then
      Writeln('Error at epoch ', i, ': ', J);

    // first order derivative of error with respect to theta
    Err := X.T.Dot(yHat - y);
    dJTheta := ((1 / m) * (((yHat - y).T.dot(X))));

    // update parameter theta
    theta := Subtract(theta, (learningRate * dJTheta));
  end;

  WriteLn;
  WriteLn('Done. Final theta:');
  PrintMatrix(theta);

  Pairs := CopyMatrix(y);
  Pairs := InsertColumnsAt(Pairs, yHat, 0);

  WriteLn('Predicted v.s. actual:');
  PrintMatrix(Pairs);

  ReadLn;
end.
