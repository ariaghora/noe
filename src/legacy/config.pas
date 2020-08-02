{$IFDEF MSWINDOWS}
BLAS_FILENAME = 'libopenblas.dll';
{$ENDIF}
{$IFDEF UNIX}
  {$IFDEF LINUX}
    BLAS_FILENAME = 'libblas.so.3';
  {$ENDIF}
  {$IFDEF DARWIN}
    BLAS_FILENAME = 'libopenblas.dylib';
  {$ENDIF}
{$ENDIF}
