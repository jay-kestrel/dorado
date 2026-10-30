*> Dorado COBOL kernel - batched D8 direction pick (phase A).
*>
*> Port of lagrangian_walker.random_pick over a batch of particle
*> weight rows: row-copy semantics (NaN -> 0), cumulative sum, pick
*> value u*total, then a searchsorted(left) scan of the cutoffs.
*>
*> Every float operation goes through one C IEEE helper (cobol_fadd,
*> cobol_fmul, cobol_fcmp) so results are bit-identical to NumPy's
*> cumsum + searchsorted path; COMP-2 COMPUTE is never used for float
*> arithmetic because GnuCOBOL lowers it to the GMP decimal runtime,
*> which rounds intermediates below binary64 precision. Integer
*> COMPUTE on COMP-5/BINARY-LONG fields is exact and used for indexing.
*>
*> Uniforms arrive pre-drawn by the Python adapter from the reference
*> MT19937 stream (port contract C5: no RNG in COBOL). The adapter
*> guarantees nw <= 64 (the D8 window is 9); WS-CUTOFFS is sized for
*> that bound.
       IDENTIFICATION DIVISION.
       PROGRAM-ID. PARTICLE-PICK.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-P PIC S9(9) COMP-5.
       01 WS-J PIC S9(9) COMP-5.
       01 WS-IDX PIC S9(9) COMP-5.
       01 WS-FOUND PIC 9.
       01 WS-CUM COMP-2.
       01 WS-TOTAL COMP-2.
       01 WS-PICK COMP-2.
       01 WS-V COMP-2.
       01 WS-CUTOFFS COMP-2 OCCURS 64 TIMES.
       01 WS-CMP PIC S9(9) COMP-5.
       LINKAGE SECTION.
       01 LS-K USAGE BINARY-LONG.
       01 LS-NW USAGE BINARY-LONG.
       01 LS-N USAGE BINARY-LONG.
       01 LS-ROWBUF.
          05 LS-ROW USAGE COMP-2 OCCURS 1 TO 268435455
                         DEPENDING ON LS-N.
       01 LS-UBUF.
          05 LS-U USAGE COMP-2 OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-OBUF.
          05 LS-OIDX USAGE BINARY-LONG OCCURS 1 TO 268435455
                         DEPENDING ON LS-K.
       PROCEDURE DIVISION USING LS-K LS-NW LS-N LS-ROWBUF LS-UBUF
                                LS-OBUF.
       MAIN.
           PERFORM VARYING WS-P FROM 1 BY 1 UNTIL WS-P > LS-K
               COMPUTE WS-IDX = (WS-P - 1) * LS-NW
               MOVE 0 TO WS-CUM
               PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > LS-NW
                   COMPUTE WS-IDX = (WS-P - 1) * LS-NW + WS-J
                   MOVE LS-ROW(WS-IDX) TO WS-V
                   *> NaN -> 0: a COMP-2 field is unequal to itself
                   *> exactly when its payload is a NaN (probe-validated)
                   IF WS-V NOT = WS-V
                       MOVE 0 TO WS-V
                   END-IF
                   IF WS-J = 1
                       MOVE WS-V TO WS-CUM
                   ELSE
                       CALL "cobol_fadd" USING WS-CUM WS-V WS-CUM
                   END-IF
                   MOVE WS-CUM TO WS-CUTOFFS(WS-J)
               END-PERFORM
               MOVE WS-CUM TO WS-TOTAL
               CALL "cobol_fmul" USING LS-U(WS-P) WS-TOTAL WS-PICK
               COMPUTE LS-OIDX(WS-P) = LS-NW - 1
               *> searchsorted-left: first cutoff >= pick; pick > every
               *> finite cutoff (u>=1 float artifacts, all-zero row)
               *> falls through to the last slot, matching NumPy
               MOVE 0 TO WS-FOUND
               PERFORM VARYING WS-J FROM 1 BY 1
                       UNTIL WS-J > LS-NW OR WS-FOUND = 1
                   CALL "cobol_fcmp" USING WS-CUTOFFS(WS-J) WS-PICK
                                           WS-CMP
                   IF WS-CMP >= 0
                       COMPUTE LS-OIDX(WS-P) = WS-J - 1
                       MOVE 1 TO WS-FOUND
                   END-IF
               END-PERFORM
           END-PERFORM
      *> The helper CALLs land their int returns in RETURN-CODE; zero
      *> it explicitly so the ctypes shim sees 0 on success.
           MOVE 0 TO RETURN-CODE
           GOBACK.
       END PROGRAM PARTICLE-PICK.
