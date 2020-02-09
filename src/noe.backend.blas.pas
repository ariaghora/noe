{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit provides an interface to OpenBLAS library.
 Dependency:
 o Linux:
   - Debian/Ubuntu/Kali: apt install libopenblas-base
 o Windows:
   - Provide the libopenblas.dll
 o OSX:
   - Provide the libopenblas.dylib
}

unit noe.backend.blas;

{$mode objfpc}{$H+}

interface

uses
  Classes, dynlibs, noe, noe.utils, SysUtils;

type
  Pdouble = ^double;

  CBLAS_ORDER     = (CblasRowMajor = 101, CblasColMajor = 102);
  CBLAS_TRANSPOSE = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113);
  CBLAS_UPLO      = (CblasUpper = 121, CblasLower = 122);
  CBLAS_DIAG      = (CblasNonUnit = 131, CblasUnit = 132);
  LAPACK_ORDER    = (LAPACKRowMajor = 101, LAPACKColMajor = 102);

  TFuncDaxpy = procedure(N: longint; Alpha: double; X: TFloatVector;
    INCX: longint; Y: TFloatVector; INCY: longint); cdecl;
  TFuncDgemm = procedure(Order: CBLAS_ORDER; TransA: CBLAS_TRANSPOSE;
    TransB: CBLAS_TRANSPOSE; M: longint; N: longint; K: longint;
    alpha: double; A: TFloatVector; lda: longint; B: TFloatVector;
    ldb: longint; beta: double; C: TFloatVector; ldc: longint);

{$IFDEF FPC}
{$PACKRECORDS C}
{$ENDIF}

var
  blas_dgemm: TFuncDgemm;
  blas_daxpy: TFuncDaxpy;

  libHandle: THandle = dynlibs.NilHandle;

function Add_BLAS(A, B: TTensor): TTensor;
function MatMul_BLAS(A, B: TTensor): TTensor;
function MeanCol_BLAS(A: TTensor): TTensor;
function MeanRow_BLAS(A: TTensor): TTensor;
function SumCol_BLAS(A: TTensor): TTensor;
function SumRow_BLAS(A: TTensor): TTensor;

implementation

uses
  noe.Math;

function Add_BLAS(A, B: TTensor): TTensor;
begin
  Assert(A.Size = B.Size, MSG_ASSERTION_DIFFERENT_LENGTH);
  Result     := CreateEmptyTensor(A.Shape);
  Result.Val := copy(B.Val);
  blas_daxpy(A.Size, 1, A.Val, 1, Result.Val, 1);
end;

function MatMul_BLAS(A, B: TTensor): TTensor;
begin
  Result := CreateEmptyTensor([A.Shape[0], B.Shape[1]]);
  blas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    A.Shape[0], B.Shape[1], B.Shape[0], // m, n, k
    1, // alpha
    A.val, B.Shape[0],
    B.val, B.Shape[1],
    1, // beta
    Result.val, B.Shape[1]
    );
  Result.ReshapeInplace([A.Shape[0], B.Shape[1]]);
end;

function MeanCol_BLAS(A: TTensor): TTensor;
begin
  Result := MatMul_BLAS(CreateTensor([1, A.Shape[0]], 1 / A.Shape[0]), A);
end;

function MeanRow_BLAS(A: TTensor): TTensor;
begin
  Result := MatMul_BLAS(A, CreateTensor([A.Shape[1], 1], 1 / A.Shape[1]));
end;

function SumCol_BLAS(A: TTensor): TTensor;
begin
  Result := MatMul_BLAS(Ones([1, A.Shape[0]]), A);
end;

function SumRow_BLAS(A: TTensor): TTensor;
begin
  Result := MatMul_BLAS(A, Ones([A.Shape[1], 1]));
end;

initialization
  libHandle := LoadLibrary(BLAS_FILENAME);

  Pointer(blas_dgemm) := (GetProcedureAddress(libHandle, 'cblas_dgemm'));
  Pointer(blas_daxpy) := (GetProcedureAddress(libHandle, 'cblas_daxpy'));

  if IsConsole then
  begin
    if blas_dgemm = nil then
      WriteLn('blas_dgemm is not supported');
    if blas_daxpy = nil then
      WriteLn('blas_daxpy is not supported');
  end;

finalization
  blas_dgemm := nil;
  blas_daxpy := nil;

end.
