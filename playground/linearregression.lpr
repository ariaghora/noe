program linearregression;

{
  This program demonstrates the usage of darkteal library to perform linear
  regression based on gradient descent algorithm for optimization.
}

uses
  wincrt,
  DTCommon,
  DTLinAlg,
  DTMLUtils;

var
  dataset: TFloatMatrix;
  X, y, yHat: TFloatMatrix;
  theta, dJTheta: TFloatMatrix;
  i, m: integer;
  learningRate, J: real;
  intercept: TFloatVector;

  test: TFloatMatrix;

  function MSE(yHat, y: TFloatMatrix): real;
  begin
    Result := (1 / m) * sum(ElementWise(@pow, 2, Subtract(yHat, y)));
  end;

begin
  // load toy dataset
  dataset := FloatMatrixFromCSV('ex2.csv');

  X := GetColumn(dataset, 0); // independent variable
  y := GetColumn(dataset, 1); // target

  // get number of samples
  m := Shape(X)[0];

  // Add intercept term to X
  intercept := CreateVector(m, 1);
  InsertColumnAt(X, 0, intercept);

  // model parameter, 2x1 matrix initialized with 0.5
  theta := CreateMatrix(2, 1, 0.5);

  // set the learning rate
  learningRate := 0.003;

  // start gradient descent
  for i := 1 to 100 do
  begin
    yHat := DotProduct(X, theta); // make prediction

    // calculate mean squared error loss
    J := MSE(yHat, y);

    if i mod 5 = 0 then
      Writeln('Error at epoch ', i, ': ', J);

    // calculate first order derivative of J with respect to theta
    // ∇J = (2/m) * sum((ŷ - y) * X))
    dJTheta := Multiply(2 / m,
      sum(Multiply(Transpose(Subtract(yHat, y)), Transpose(X)), 1));

    // update parameter theta
    theta := Subtract(theta, Multiply(learningRate, dJTheta));
  end;

  WriteLn('Done');
  ReadLn;
end.
