unit DTMLUtils;

{ A unit containing machine-learning related classes anc functions }

{$mode delphi}

interface

uses
  Classes, SysUtils, Math, DTCore, DTLinAlg;

function Sigmoid(x: double): double; overload;
function Sigmoid(A: TDTMatrix): TDTMatrix; overload;
function SigmoidPrime(x: double): double; overload;
function SigmoidPrime(A: TDTMatrix): TDTMatrix; overload;
function Relu(x: double): double;
function ReluPrime(x: double): double;

implementation

{ @abstract(Sigmoid activation function) }
function Sigmoid(x: double): double;
begin
  Result := 1 / (1 + exp(-x));
end;

function Sigmoid(A: TDTMatrix): TDTMatrix;
var
  i: integer;
begin
  Result := Apply(@Sigmoid, A);
end;

function SigmoidPrime(x: double): double;
begin
  Result := Sigmoid(x) * (1 - Sigmoid(x));
end;

function SigmoidPrime(A: TDTMatrix): TDTMatrix;
begin
  Result := Apply(@SigmoidPrime, A);
end;

function Relu(x: double): double;
begin
  Result := x * integer(x > 0);
end;

function ReluPrime(x: double): double;
begin
  Result := 1 * integer(x > 0);
end;


end.

