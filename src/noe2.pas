unit noe2;

{$mode objfpc}{$H+}
{$modeswitch advancedRecords}

interface

uses
  Classes, SysUtils, multiarray, numerik, fgl;

type

  TTensor = class
    Data: TMultiArray;
    BackwardFunc: Pointer;
    Deps: array of TTensor;
    IsLeaf: boolean;
  private
    FGrad: TMultiArray;
    FRequiresGrad: boolean;
    function GetGrad: TMultiArray;
    function GetShape: TLongVector;
    procedure AddDependencies(ADeps: array of TTensor);
    procedure SetRequiresGrad(val: boolean);
  public
    destructor Destroy; override;
    procedure Backward(G: TMultiArray);
    procedure ZeroGrad;
    property Grad: TMultiArray read GetGrad write FGrad;
    property RequiresGrad: boolean read FRequiresGrad write SetRequiresGrad;
    property Shape: TLongVector read GetShape;
  end;

  TBackwardFunc = procedure(var arr: array of TTensor; G: TMultiArray);
  TTensorList = specialize TFPGObjectList<TTensor>;

procedure PrintTensor(T: TTensor);

function CreateTensor(Data: TMultiArray; RequiresGrad: boolean=False): TTensor;

function Add(A, B: TTensor): TTensor; overload;
function Matmul(A, B: TTensor): TTensor; overload;
function Mean(A: TTensor): TTensor; overload;
function Multiply(A, B: TTensor): TTensor; overload;
function Negate(A: TTensor): TTensor; overload;
function ReLU(A: TTensor): TTensor; overload;
function Subtract(A, B: TTensor): TTensor; overload;
function Sqr(A: TTensor): TTensor; overload;
function Sum(A: TTensor): TTensor; overload;

operator :=(A: TMultiArray) B: TTensor;

var
  NoeGlobalTensorList: TTensorList;

implementation

procedure TTensor.AddDependencies(ADeps: array of TTensor);
var
  i: integer;
begin
  SetLength(Deps, Length(ADeps));
  for i := 0 to High(ADeps) do
  begin
    Self.RequiresGrad := Self.RequiresGrad or ADeps[i].RequiresGrad;
    Deps[i] := ADeps[i];
  end;
end;

procedure TTensor.SetRequiresGrad(val: boolean);
begin
  self.FRequiresGrad := val;
  if val then
    self.Grad := Zeros(Self.Data.Shape)

end;

function TopologicalSort(T: TTensor): TTensorList;
var
  Seen, Sorted: TTensorList;
  prv: TTensor;

  procedure TopoHelper(v: TTensor);
  begin
    if (Seen.IndexOf(v) = -1) then
    begin
      Seen.Add(v);
      for prv in v.Deps do
        TopoHelper(prv);

      if v.RequiresGrad then
        Sorted.Add(v);
    end;
  end;

begin
  Seen := TTensorList.Create(False);
  Sorted := TTensorList.Create(False);
  TopoHelper(T);

  Result := Sorted;
  Seen.Free;
end;


procedure TTensor.Backward(G: TMultiArray);
var
  i: integer;
  Sorted: TTensorList;
begin
  if not self.RequiresGrad then
    raise Exception.Create('Cannot call backward on tensor not requiring grad.');
  if not VectorEqual(self.Shape, G.Shape) then
    raise Exception.Create('G must have the same dimension.');

  Sorted := TopologicalSort(self);
  self.Grad := G;

  for i := Sorted.Count - 1 downto 0 do
  begin
    if Assigned(Sorted[i].BackwardFunc) then
    begin
      TBackwardFunc(Sorted[i].BackwardFunc)(Sorted[i].Deps, Sorted[i].Grad);
    end;
  end;

  { Remove the unused Tensors in the previous pass }
  for i := NoeGlobalTensorList.Count - 1 downto 0 do
    if (Sorted.IndexOf(NoeGlobalTensorList[i]) = -1) and not(NoeGlobalTensorList[i].IsLeaf) then
      NoeGlobalTensorList.Remove(NoeGlobalTensorList[i]);

  Sorted.free;
end;

procedure TTensor.ZeroGrad;
begin
  if not RequiresGrad then Exit;
  Grad := Zeros(self.Shape);
end;

destructor TTensor.Destroy;
begin
  self.Deps := nil;
end;

procedure PrintTensor(T: TTensor);
begin
  PrintMultiArray(T.Data);
end;

function CreateTensor(Data: TMultiArray; RequiresGrad: boolean=False): TTensor;
begin
  Result := TTensor.Create;
  Result.RequiresGrad := RequiresGrad;
  Result.Data := Data;
  Result.BackwardFunc := nil;
  Result.IsLeaf := True;
  NoeGlobalTensorList.Add(Result);
end;

function CreateOpNode(Val: TTensor; Deps: array of TTensor; BackwardFunc: TBackwardFunc): TTensor;
begin
  Result := Val;
  Result.AddDependencies(Deps);
  Result.BackwardFunc := BackwardFunc;
  Result.IsLeaf := False;
end;

function ReduceGradToShape(Grad: TMultiArray; Shape: TLongVector): TMultiArray;
var
  i, NDimsAdded: integer;
begin
  NDimsAdded := Grad.NDims - Length(Shape);
  for i := 0 to NDimsAdded - 1 do
    Grad := Sum(Grad, 0);

  for i := 0 to High(Shape) do
    if Shape[i] = 1 then
      Grad := Sum(Grad, i, True);
  Result := Grad;

end;

procedure AddBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + ReduceGradToShape(G, Deps[0].Shape);
  if Deps[1].RequiresGrad then
    Deps[1].Grad := Deps[1].Grad + ReduceGradToShape(G, Deps[1].Shape);
end;

function Add(A, B: TTensor): TTensor;
begin
  Exit(CreateOpNode(A.Data + B.Data, [A, B], @AddBackward));
end;

procedure MatmulBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + G.Matmul(Deps[1].Data.T);
  if Deps[1].RequiresGrad then
    Deps[1].Grad := Deps[1].Grad + Deps[0].Data.T.Matmul(G);
end;

function Matmul(A, B: TTensor): TTensor;
begin
  Exit(CreateOpNode(A.Data.Matmul(B.Data), [A, B], @MatmulBackward));
end;

procedure MultiplyBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + ReduceGradToShape(G * Deps[1].Data, Deps[0].Shape);
  if Deps[1].RequiresGrad then
    Deps[1].Grad := Deps[1].Grad + ReduceGradToShape(G * Deps[0].Data, Deps[1].Shape);
end;

procedure MeanBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + FullMultiArray(Deps[0].Data.Shape,
      G.Data[0] / (Deps[0].Data.Size / G.Size));
end;

function Mean(A: TTensor): TTensor;
begin
  Exit(CreateOpNode(Mean(A.Data), [A], @MeanBackward));
end;

function Multiply(A, B: TTensor): TTensor;
begin
  Exit(CreateOpNode(A.Data * B.Data, [A, B], @MultiplyBackward));
end;

procedure NegateBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad - G;
end;

function Negate(A: TTensor): TTensor;
begin
  Exit(CreateOpNode(-A.Data, [A], @NegateBackward));
end;

procedure ReLUBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + (G > 0);
end;

function ReLU(A: TTensor): TTensor; overload;
begin
  Exit(CreateOpNode(Maximum(A.Data, 0), [A], @ReLUBackward));
end;

procedure SubtractBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + ReduceGradToShape(G, Deps[0].Shape);
  if Deps[1].RequiresGrad then
    Deps[1].Grad := Deps[1].Grad - ReduceGradToShape(G, Deps[1].Shape);
end;

function Subtract(A, B: TTensor): TTensor;
begin
  Exit(CreateOpNode(A.Data - B.Data, [A, B], @SubtractBackward));
end;

procedure SumBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + Ones(Deps[0].Grad.Shape);
end;

procedure SqrBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + (2 * G * Deps[0].Data);
end;

function Sqr(A: TTensor): TTensor;
begin
  Exit(CreateOpNode((A.Data ** 2), [A], @SqrBackward));
end;

function Sum(A: TTensor): TTensor;
begin
  Exit(CreateOpNode(Sum(A.Data), [A], @SumBackward));
end;

operator +(A, B: TTensor)C: TTensor;
begin
  C := Add(A, B);
end;

function TTensor.GetGrad: TMultiArray;
begin
  if RequiresGrad then Exit(FGrad);
  raise Exception.Create('Trying to access Grad of a tensor that has no Grad.');
end;

function TTensor.GetShape: TLongVector;
begin
  Exit(Self.Data.Shape);
end;

operator :=(A: TMultiArray) B: TTensor;
begin
  B := CreateTensor(A);
end;

initialization
  NoeGlobalTensorList := TTensorList.Create;

finalization
  NoeGlobalTensorList.Free;

end.

