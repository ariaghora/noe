{ A unit providing some linear algebra functionalities }
unit DTLinAlg;

{$mode delphi}

interface

uses
  Classes, SysUtils, Math, DTCore;

type

  { @abstract(A class to compute principal component analysis) }
  TPCA = class
    k: integer;
    { The number of principal components }
    NComponents: integer;
    Components: TDTMatrix;
  public
    { Start computing principal components. The current implementation is
      based on eigendecomposition of the covariance (mean-centered) matrix }
    constructor Create(NComponents: integer);
    function Fit(X: TDTMatrix): TPCA;
    { Transform X into NComponents principal components }
    function Transform(X: TDTMatrix): TDTMatrix;
  end;

{ @abstract(Calculate SVD of X)

  Store the result in U, Sigma, and VT respectively. }
procedure SVD(X: TDTMatrix; var U: TDTMatrix; var Sigma: TDTMatrix; var VT: TDTMatrix);

implementation

procedure SVD(X: TDTMatrix; var U: TDTMatrix; var Sigma: TDTMatrix; var VT: TDTMatrix);
var
  Superb: TFloatVector;
begin
  SetLength(Superb, Min(X.Width, X.Height) - 1);
  SetLength(Sigma.val, X.Width);
  SetLength(U.val, X.Height * X.Height);
  SetLength(VT.val, X.Width * X.Width);

  Sigma.Width := X.Width;
  Sigma.Height := 1;
  U.Width := X.Height;
  U.Height := X.Height;
  VT.Width := X.Width;
  VT.Height := X.Width;

  LAPACKE_dgesvd(LAPACKRowMajor, 'A', 'A', X.Height, X.Width, CopyMatrix(X).val,
    X.Width, Sigma.val, U.val, X.Height, VT.val, X.Width, Superb);
end;


constructor TPCA.Create(NComponents: integer);
begin
  self.NComponents := NComponents;
end;

function TPCA.Fit(X: TDTMatrix): TPCA;
var
  Xc, C, VL, VR, WR: TDTMatrix;
  wi: TFloatVector;
  EigTmp: double;
  i, j: integer;
begin
  { Mean-center the input matrix }
  Xc := X - Mean(X, 0);

  { Compute the covariance }
  C := Xc.T.Dot(Xc) / (Xc.Width);

  { Run eigendecomposition }
  SetLength(WR.val, C.Height);
  SetLength(wi, C.Height);
  SetLength(VL.val, C.Height * C.Height);
  SetLength(VR.val, C.Height * C.Height);
  VL.Width := C.Height;
  VL.Height := C.Height;
  VR.Width := C.Height;
  VR.Height := C.Height;
  WR.Height := 1;
  WR.Width := C.Height;

  LAPACKE_dgeev(LAPACKRowMajor, 'V', 'V', C.Height, CopyMatrix(C).val,
    C.Height, WR.val, wi,
    VL.val, C.Height, VR.val, C.Height);

  { Sort the eigenvalues in nondecreasing order }
  for i := 0 to High(WR.val) do
    for j := 0 to High(WR.val) do
    begin
      if WR.val[j] < WR.val[i] then
      begin
        EigTmp := WR.val[j];
        WR.val[j] := WR.val[i];
        WR.val[i] := EigTmp;
        SwapColumns(VR, i, j);
      end;
    end;
  VR := VR.GetRange(0, 0, VR.Height, NComponents);
  Components := VR.T;
  //X_pca := VR.T.Dot(Xc.T).T;
  Result := self;
end;

function TPCA.Transform(X: TDTMatrix): TDTMatrix;
var
  Xc: TDTMatrix;
begin
  Xc := X - Mean(X, 0);
  Result := Components.Dot(Xc.T).T;
end;

end.
