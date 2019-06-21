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
  public
    function Fit(X: TDTMatrix): TPCA;
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

{ TPCA }

function TPCA.Fit(X: TDTMatrix): TPCA;
var
  U, Sigma, S, VT, V, Xc: TDTMatrix;
begin
  SVD(X, U, Sigma, VT);

  PrintMatrix(U);
  PrintMatrix(Sigma);
  PrintMatrix(VT);
  PrintMatrix(X.Dot(VT));

end;

end.
