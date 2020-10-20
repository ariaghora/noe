unit noe;

{$mode objfpc}{$H+}
{$modeswitch advancedRecords}

interface

uses
  Math, SysUtils, multiarray, numerik, fgl;

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
    function GetItems(idx: array of TLongVector): TMultiArray;
    function GetShape: TLongVector;
    procedure AddDependencies(ADeps: array of TTensor);
    procedure SetRequiresGrad(val: boolean);
  public
    destructor Destroy; override;
    function Matmul(T: TTensor): TTensor;
    procedure Backward;
    procedure Backward(G: TMultiArray);
    procedure ZeroGrad;
    property Grad: TMultiArray read GetGrad write FGrad;
    property Items[idx: array of TLongVector]: TMultiArray read GetItems; default;
    property RequiresGrad: boolean read FRequiresGrad write SetRequiresGrad;
    property Shape: TLongVector read GetShape;
  end;

  TBackwardFunc = procedure(var arr: array of TTensor; G: TMultiArray);
  TTensorList = specialize TFPGObjectList<TTensor>;

procedure PrintTensor(T: TTensor);

function CreateTensor(Data: TMultiArray; RequiresGrad: boolean = False): TTensor;
function BinarizeLabel(T: TTensor): TTensor;

function Add(A, B: TTensor): TTensor; overload;
function Conv2d(X, W: TTensor; Stride, Pad: longint): TTensor;
function Divide(A, B: TTensor): TTensor; overload;
function Exp(A: TTensor): TTensor; overload;
function LeakyReLU(A: TTensor; Leakiness: single): TTensor; overload;
function Ln(A: TTensor): TTensor; overload;
function Matmul(A, B: TTensor): TTensor; overload;
function Max(A: TTensor; axis: integer = -1; KeepDims: boolean = False): TTensor; overload;
function Mean(A: TTensor; axis: integer = -1; KeepDims: boolean = False): TTensor; overload;
function Multiply(A, B: TTensor): TTensor; overload;
function Negate(A: TTensor): TTensor; overload;
function ReLU(A: TTensor): TTensor; overload;
function Sigmoid(A: TTensor): TTensor; overload;
function Softmax(A: TTensor; axis: integer): TTensor; overload;
function Subtract(A, B: TTensor): TTensor; overload;
function Sqr(A: TTensor): TTensor; overload;
function Sum(A: TTensor; axis: integer = -1; KeepDims: boolean = False): TTensor; overload;

{ Cross entropy loss, with ground truth represented as one-hot matrix }
function CrossEntropy(YPred, Y: TTensor; Tol: single=1e-8): TTensor;

{ @exclude } operator +(A, B: TTensor) C: TTensor;
{ @exclude } operator -(A: TTensor) B: TTensor;
{ @exclude } operator -(A, B: TTensor) C: TTensor;
{ @exclude } operator * (A, B: TTensor) C: TTensor;
{ @exclude } operator / (A, B: TTensor) C: TTensor;
{ @exclude } operator := (A: TMultiArray) B: TTensor;
{ @exclude } operator := (A: single) B: TTensor;
{ @exclude } operator := (A: TTensor) B: TMultiArray;

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
    self.Grad := Zeros(Self.Data.Shape);

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
    if (Sorted.IndexOf(NoeGlobalTensorList[i]) = -1) and not
      (NoeGlobalTensorList[i].IsLeaf) then
      NoeGlobalTensorList.Remove(NoeGlobalTensorList[i]);

  Sorted.Free;
end;

procedure TTensor.ZeroGrad;
begin
  if not RequiresGrad then
    Exit;
  Grad := Zeros(self.Shape);
end;

destructor TTensor.Destroy;
begin
  self.Deps := nil;
end;

function TTensor.Matmul(T: TTensor): TTensor;
begin
  Exit(noe.Matmul(Self, T));
end;

procedure TTensor.Backward;
begin
  self.Backward(1);
end;

procedure PrintTensor(T: TTensor);
begin
  PrintMultiArray(T.Data);
end;

function BinarizeLabel(T: TTensor): TTensor;
var
  MaxVal: single;
  i: longint;
begin
  if T.Data.Squeeze.NDims > 1 then
    raise Exception.Create('Can only accept a tensor with NDim=1 or a column tensor');
  MaxVal := MaxValue(T.Data.Data);
  Result := Zeros([T.Data.Size, Round(MaxVal) + 1]);
  for i := 0 to Result.Data.Shape[0] - 1 do
    Result.Data.Put([i, Round(T.Data.Get(i))], 1);
end;

function CreateTensor(Data: TMultiArray; RequiresGrad: boolean = False): TTensor;
begin
  Result := TTensor.Create;
  Result.RequiresGrad := RequiresGrad;
  Result.Data := Data;
  Result.BackwardFunc := nil;
  Result.IsLeaf := True;
  NoeGlobalTensorList.Add(Result);
end;

function CreateOpNode(Val: TTensor; Deps: array of TTensor;
  BackwardFunc: TBackwardFunc): TTensor;
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

procedure DivideBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + ReduceGradToShape(G / Deps[1].Data, Deps[0].Shape);
  if Deps[1].RequiresGrad then
    Deps[1].Grad := Deps[1].Grad + ReduceGradToShape(-G * Deps[0].Data /
      Deps[1].Data ** 2, Deps[1].Shape);
end;

procedure Conv2dBackward(var Deps: array of TTensor; G: TMultiArray);
var
  GFlat: TMultiArray;
  TrailingDim: longint;
begin
  //if Deps[0].RequiresGrad then
  //  Deps[0].Grad;
  if Deps[1].RequiresGrad then
  begin
    PrintMultiArray(TSingleVector(G.Shape));
    TrailingDim := G.Shape[1] * G.Shape[2] * G.Shape[3];
    GFlat := Transpose(G, [1, 2, 3, 0]).Reshape([G.Shape[0], TrailingDim]);
    Deps[1].Grad := Deps[1] + GFlat.Matmul(Deps[0].Data.T).Reshape(Deps[1].Shape);
  end;
end;

function Conv2d(X, W: TTensor; Stride, Pad: longint): TTensor;
var
  deps: array of TTensor;
  OutH, OutW: longint;
  ConvRes, WCol, XCols: TMultiArray;
begin
  OutH := (X.Shape[2] - W.Shape[2]) div Stride + 1;
  OutW := (X.Shape[3] - W.Shape[3]) div Stride + 1;
  WCol := W.Data.Reshape([W.Shape[0], W.Shape[1] * W.Shape[2] * W.Shape[3]]);

  XCols := Im2Col(X.Data, W.Shape[2], w.Shape[3], Stride, Pad);

  ConvRes := WCol.matmul(XCols);
  ConvRes := ConvRes.Reshape([X.Shape[0], W.Shape[0], OutH, OutW]).Contiguous;

  deps := [XCols, W, Stride, Pad];
  Exit(CreateOpNode(ConvRes, deps, @Conv2dBackward));
end;

function Divide(A, B: TTensor): TTensor;
begin
  Exit(CreateOpNode(A.Data / B.Data, [A, B], @DivideBackward));
end;

procedure ExpBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + (G * Exp(Deps[0].Data));
end;

function Exp(A: TTensor): TTensor; overload;
begin
  Exit(CreateOpNode(Exp(A.Data), [A], @ExpBackward));
end;

procedure LnBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + (G / Deps[0].Data);
end;

procedure LeakyReLUBackward(var Deps: array of TTensor; G: TMultiArray);
var
  i: longint;
begin
  if Deps[0].RequiresGrad then
    for i := 0 to Deps[0].Data.Size - 1 do
      if Deps[0].Data.Get(i) > 0 then
        Deps[0].Grad.Data[i] := Deps[0].Grad.Data[i] + G.Data[i]
      else
        { Deps[1].Data.Get(0) refers to 'Leakiness' parameter in LeakyReLU }
        Deps[0].Grad.Data[i] := Deps[0].Grad.Data[i] + G.Data[i] * Deps[1].Data.Get(0);
end;

function LeakyReLU(A: TTensor; Leakiness: single): TTensor;
var
  OutArr: TMultiArray;
  i: integer;
  v: single;
begin
  OutArr := AllocateMultiArray(A.Data.Size).Reshape(A.Shape);
  for i := 0 to A.Data.Size - 1 do
  begin
    v := A.Data.Get(i);
    OutArr.Data[i] := IfThen(v < 0, v * Leakiness, v);
  end;
  Exit(CreateOpNode(OutArr, [A, TMultiArray(Leakiness)], @LeakyReluBackward));
end;

function Ln(A: TTensor): TTensor;
begin
  Exit(CreateOpNode(Ln(A.Data), [A], @LnBackward));
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
var
  Shape: TLongVector;
  Axis: integer;
begin
  if Deps[0].RequiresGrad then
  begin
    { If axis is specified, then G should be reshaped accordingly to comply
      with broadcasting. }
    Axis := Round(Deps[1].Data.Data[0]);
    if Axis > -1 then
    begin
      Shape := CopyVector(Deps[0].Shape);
      Shape[Axis] := 1;
      G := G.Reshape(Shape) / Deps[0].Shape[Axis];
    end
    else
    begin
      G := G.Item / Deps[0].Data.Size;
    end;

    Deps[0].Grad := Deps[0].Grad + G;
  end;
end;

function Mean(A: TTensor; axis: integer = -1; KeepDims: boolean = False): TTensor;
begin
  Exit(CreateOpNode(Mean(A.Data, axis, KeepDims), [A, TMultiArray(axis), TMultiArray(integer(KeepDims))],
    @MeanBackward));
end;

procedure MaxBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + (Deps[0].Data = Deps[1].Data) *
      G.Reshape(Deps[1].Shape);
end;

function Max(A: TTensor; axis: integer; KeepDims: boolean = False): TTensor;
var
  tmp1, tmp2: TMultiArray;
begin
  tmp1 := Max(A.Data, axis, KeepDims);
  tmp2 := tmp1.Copy();
  if not KeepDims then
    SqueezeMultiArrayAt(tmp2, axis);
  Exit(CreateOpNode(tmp2, [A, tmp1], @MaxBackward));
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
var
  i: longint;
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + G * (Deps[0].Data > 0);
end;

function ReLU(A: TTensor): TTensor; overload;
begin
  Exit(CreateOpNode(Maximum(A.Data, 0), [A], @ReLUBackward));
end;

function _Sigmoid(A: TMultiArray): TMultiArray;
begin
  Exit(1/(1 + Exp(-A)));
end;

procedure SigmoidBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + _Sigmoid(Deps[0].Data) * (1 - _Sigmoid(Deps[0].Data));
end;

function Sigmoid(A: TTensor): TTensor; overload;
begin
  Exit(CreateOpNode(_Sigmoid(A.Data), [A], @SigmoidBackward));
end;

procedure SubtractBackward(var Deps: array of TTensor; G: TMultiArray);
begin
  if Deps[0].RequiresGrad then
    Deps[0].Grad := Deps[0].Grad + ReduceGradToShape(G, Deps[0].Shape);
  if Deps[1].RequiresGrad then
    Deps[1].Grad := Deps[1].Grad - ReduceGradToShape(G, Deps[1].Shape);
end;

function Softmax(A: TTensor; axis: integer): TTensor; overload;
begin
  Result := Exp(A - Max(A, axis, True));
  Result := Result / Sum(Result, axis, True);
end;

function Subtract(A, B: TTensor): TTensor;
begin
  Exit(CreateOpNode(A.Data - B.Data, [A, B], @SubtractBackward));
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

procedure SumBackward(var Deps: array of TTensor; G: TMultiArray);
var
  Shape: TLongVector;
  Axis: integer;
begin
  if Deps[0].RequiresGrad then
  begin
    { If axis is specified, then G should be reshaped accordingly to comply
      with broadcasting. }
    Axis := Round(Deps[1].Data.Get(0));
    if Axis > -1 then
    begin
      Shape := CopyVector(Deps[0].Shape);
      Shape[Axis] := 1;
      G := G.Reshape(Shape);
    end;

    Deps[0].Grad := Deps[0].Grad + G;
  end;
end;


function Sum(A: TTensor): TTensor;
begin
  Exit(CreateOpNode(Sum(A.Data), [A], @SumBackward));
end;

function Sum(A: TTensor; axis: integer; KeepDims: boolean): TTensor;
begin
  Exit(CreateOpNode(Sum(A.Data, axis, KeepDims),
    [A, TMultiArray(axis), TMultiArray(integer(KeepDims))], @SumBackward));
end;

function TTensor.GetGrad: TMultiArray;
begin
  if RequiresGrad then
    Exit(FGrad);
  raise Exception.Create('Trying to access Grad of a tensor that has no Grad.');
end;

function TTensor.GetItems(idx: array of TLongVector): TMultiArray;
begin
  Exit(Data[Idx]);
end;

function TTensor.GetShape: TLongVector;
begin
  Exit(Self.Data.Shape);
end;

function CrossEntropy(YPred, Y: TTensor; Tol: single=1e-8): TTensor;
begin
  if YPred.Data.Size <> Y.Data.Size then
    raise Exception.Create('A and B have different size.');
  Exit(-Mean(Sum(Y * Ln(YPred + Tol), 1)));
end;

operator +(A, B: TTensor)C: TTensor;
begin
  C := Add(A, B);
end;

operator -(A: TTensor) B: TTensor;
begin
  B := Negate(A);
end;

operator -(A, B: TTensor) C: TTensor;
begin
  C := Subtract(A, B);
end;

operator * (A, B: TTensor) C: TTensor;
begin
  C := Multiply(A, B);
end;

operator / (A, B: TTensor) C: TTensor;
begin
  C := Divide(A, B);
end;

operator := (A: TMultiArray) B: TTensor;
begin
  B := CreateTensor(A);
end;

operator := (A: single) B: TTensor;
begin
  B := TMultiArray(A);
end;

operator := (A: TTensor) B: TMultiArray;
begin
  B := A.Data;
end;

initialization
  NoeGlobalTensorList := TTensorList.Create;

finalization
  NoeGlobalTensorList.Free;

end.
