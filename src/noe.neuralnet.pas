unit noe.neuralnet;

{$mode objfpc}{$H+}

interface

uses
  fgl, multiarray, numerik, noe;

type
  TLayer = class
    Params: TTensorList;
  public
    constructor Create; virtual;
    destructor Destroy; override;
    function Eval(X: TTensor): TTensor; virtual; abstract;
  end;

  TLayerList = specialize TFPGObjectList<TLayer>;

  TLayerDense = class(TLayer)
  public
    constructor Create(InSize, OutSize: longint); overload;
    function Eval(X: TTensor): TTensor; override;
  end;

  TLayerLeakyReLU = class(TLayer)
  private
    FLeakiness: single;
  public
    constructor Create(Leakiness: single); overload;
    function Eval(X: TTensor): TTensor; override;
  end;

  TLayerReLU = class(TLayer)
  public
    function Eval(X: TTensor): TTensor; override;
  end;

  TLayerSoftmax = class(TLayer)
  private
    FAxis: integer;
  public
    constructor Create(Axis: integer); overload;
    function Eval(X: TTensor): TTensor; override;
  end;

  TNNModel = class
  private
    FLayerList: TLayerList;
    function GetParams: TTensorList;
  public
    FParams: TTensorList;
    constructor Create;
    destructor Destroy; override;
    procedure AddLayer(ALayer: TLayer);
    function Eval(X: TTensor): TTensor;
    property Params: TTensorList read GetParams;
  end;

implementation

constructor TLayerLeakyReLU.Create(leakiness: Single);
begin
  inherited Create;
end;

function TLayerLeakyReLU.Eval(X: TTensor): TTensor;
begin
  Exit(LeakyReLU(X, FLeakiness));
end;

{ TLayerReLU }

function TLayerReLU.Eval(X: TTensor): TTensor;
begin
  Exit(ReLU(X));
end;

constructor TLayerSoftmax.Create(Axis: integer);
begin
  inherited Create;
  Faxis := Axis;
end;

{ TLayerSoftmax }

function TLayerSoftmax.Eval(X: TTensor): TTensor;
begin
  Exit(Softmax(X, FAxis));
end;

{ TLayerDense }

constructor TLayerDense.Create(InSize, OutSize: longint);
var
  W, b: TTensor;
begin
  inherited Create;
  W := RandG(0, 1, [InSize, OutSize]) * ((2 / (InSize + OutSize)) ** 0.5);
  W.RequiresGrad := True;
  b := FullMultiArray([OutSize], 0);
  b.RequiresGrad := True;
  Params.Add(W);
  Params.Add(b);
end;

function TLayerDense.Eval(X: TTensor): TTensor;
begin
  Exit(X.Matmul(Params[0]) + Params[1]);
end;

{ TLayer }

constructor TLayer.Create;
begin
  Params := TTensorList.Create(False);
end;

destructor TLayer.Destroy;
begin
  inherited;
  Params.Free;
end;

{ TNNModel }

function TNNModel.GetParams: TTensorList;
var
  L: TLayer;
begin
  FParams.Clear;
  for L in FLayerList do
    FParams.AddList(L.Params);
  Exit(FParams);
end;

constructor TNNModel.Create;
begin
  FParams := TTensorList.Create(False);
  FLayerList := TLayerList.Create();
end;

destructor TNNModel.Destroy;
begin
  inherited;
  FLayerList.Free;
  FParams.Free;
end;

procedure TNNModel.AddLayer(ALayer: TLayer);
begin
  FLayerList.Add(ALayer);
end;

function TNNModel.Eval(X: TTensor): TTensor;
var
  L: TLayer;
begin
  Result := X;
  for L in FLayerList do
  begin
    Result := L.Eval(Result);
  end;
end;

end.
