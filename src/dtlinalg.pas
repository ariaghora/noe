{ A unit providing some linear algebra functionalities }
unit DTLinAlg;

{$mode delphi}

interface

uses
  Classes, SysUtils, Math, DTCore;

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

  LAPACKE_dgesvd(CblasRowMajor, 'A', 'A', X.Height, X.Width, CopyMatrix(X).val,
    X.Width, Sigma.val, U.val, X.Height, VT.val, X.Width, Superb);
end;

end.
