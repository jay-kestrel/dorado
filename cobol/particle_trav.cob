*> Dorado COBOL kernel - batched travel times (phase B).
*>
*> Port of lagrangian_walker.calc_travel_times' moved branch over a
*> batch of particles, in the reference's exact op order:
*>
*>   diff     = (0.5 - u) * diff_coeff * diff_mult   (when diff_coeff > 0)
*>   angle    = angle_new - angle_old
*>   proj     = dist * max(0, cos(angle))
*>   t        = 0.5*proj * (1/old_vel + 1/new_vel) * (1 + diff)
*>   t       *= velocity_mult
*>
*> Unmoved particles (boundary-frozen, deposited, parked, or zero-step)
*> get 0.0 and consume no uniform. Every float operation goes through
*> one C IEEE helper (cobol_fadd/fsub/fmul/fdiv/fcos/fcmp) so results
*> are bit-identical to NumPy's stepwise evaluation, including IEEE
*> special values (zero velocities, NaN, signed zeros); COMP-2 COMPUTE
*> is never used for float arithmetic because GnuCOBOL lowers it to the
*> GMP decimal runtime.
*>
*> The diffusion uniform arrives pre-drawn by the Python adapter from
*> the reference MT19937 stream, one per moved particle in index order
*> (port contract C5: no RNG in COBOL).
       IDENTIFICATION DIVISION.
       PROGRAM-ID. PARTICLE-TRAV.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-P PIC S9(9) COMP-5.
       01 WS-HALF COMP-2.
       01 WS-ONE COMP-2.
       01 WS-ZERO COMP-2.
       01 WS-DIFFU COMP-2.
       01 WS-DIFF COMP-2.
       01 WS-ANG COMP-2.
       01 WS-COSR COMP-2.
       01 WS-PROJ COMP-2.
       01 WS-T COMP-2.
       01 WS-I1 COMP-2.
       01 WS-I2 COMP-2.
       01 WS-SUM COMP-2.
       01 WS-G COMP-2.
       01 WS-CMP PIC S9(9) COMP-5.
       LINKAGE SECTION.
       01 LS-K USAGE BINARY-LONG.
       01 LS-DISTBUF.
          05 LS-DIST USAGE COMP-2 OCCURS 1 TO 268435455
                          DEPENDING ON LS-K.
       01 LS-OVBUF.
          05 LS-OV USAGE COMP-2 OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-NVBUF.
          05 LS-NV USAGE COMP-2 OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-ANBUF.
          05 LS-AN USAGE COMP-2 OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-AOBUF.
          05 LS-AO USAGE COMP-2 OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-DUBUF.
          05 LS-DU USAGE COMP-2 OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-DMBUF.
          05 LS-DM USAGE COMP-2 OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-VMBUF.
          05 LS-VM USAGE COMP-2 OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-MVBUF.
          05 LS-MV USAGE BINARY-LONG OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-DIFFCOEFF USAGE COMP-2.
       01 LS-OBUF.
          05 LS-OUT USAGE COMP-2 OCCURS 1 TO 268435455
                         DEPENDING ON LS-K.
       PROCEDURE DIVISION USING LS-K LS-DISTBUF LS-OVBUF LS-NVBUF
                                LS-ANBUF LS-AOBUF LS-DUBUF LS-DMBUF
                                LS-VMBUF LS-MVBUF LS-DIFFCOEFF
                                LS-OBUF.
       MAIN.
           MOVE 0.5 TO WS-HALF
           MOVE 1.0 TO WS-ONE
           MOVE 0.0 TO WS-ZERO
           PERFORM VARYING WS-P FROM 1 BY 1 UNTIL WS-P > LS-K
               IF LS-MV(WS-P) = 0
                   *> particle did not move: the reference leaves the
                   *> Python int 0 here, the adapter re-types it
                   MOVE WS-ZERO TO LS-OUT(WS-P)
               ELSE
                   *> diffusion: diff_coeff > 0 consumes one uniform
                   CALL "cobol_fcmp" USING LS-DIFFCOEFF WS-ZERO WS-CMP
                   IF WS-CMP > 0
                       MOVE LS-DU(WS-P) TO WS-DIFFU
                       CALL "cobol_fsub" USING WS-HALF WS-DIFFU WS-DIFF
                       CALL "cobol_fmul" USING WS-DIFF LS-DIFFCOEFF
                                               WS-DIFF
                       CALL "cobol_fmul" USING WS-DIFF LS-DM(WS-P)
                                               WS-DIFF
                   ELSE
                       MOVE WS-ZERO TO WS-DIFF
                   END-IF
                   *> direction cosine factor; max(0, cos) clamps
                   *> backwards/orthogonal steps to +0.0 (NaN included,
                   *> matching Python's max(0, nan) -> 0)
                   CALL "cobol_fsub" USING LS-AN(WS-P) LS-AO(WS-P)
                                           WS-ANG
                   CALL "cobol_cos" USING WS-ANG WS-COSR
                   CALL "cobol_fcmp" USING WS-COSR WS-ZERO WS-CMP
                   IF WS-CMP > 0
                       CALL "cobol_fmul" USING LS-DIST(WS-P) WS-COSR
                                               WS-PROJ
                   ELSE
                       MOVE WS-ZERO TO WS-PROJ
                   END-IF
                   *> average-velocity travel time
                   CALL "cobol_fmul" USING WS-HALF WS-PROJ WS-T
                   CALL "cobol_fdiv" USING WS-ONE LS-OV(WS-P) WS-I1
                   CALL "cobol_fdiv" USING WS-ONE LS-NV(WS-P) WS-I2
                   CALL "cobol_fadd" USING WS-I1 WS-I2 WS-SUM
                   CALL "cobol_fmul" USING WS-T WS-SUM WS-T
                   CALL "cobol_fadd" USING WS-ONE WS-DIFF WS-G
                   CALL "cobol_fmul" USING WS-T WS-G WS-T
                   CALL "cobol_fmul" USING WS-T LS-VM(WS-P) WS-T
                   MOVE WS-T TO LS-OUT(WS-P)
               END-IF
           END-PERFORM
      *> The helper CALLs land their int returns in RETURN-CODE; zero
      *> it explicitly so the ctypes shim sees 0 on success.
           MOVE 0 TO RETURN-CODE
           GOBACK.
       END PROGRAM PARTICLE-TRAV.
