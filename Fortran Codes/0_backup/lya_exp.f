        IMPLICIT NONE
        DOUBLE PRECISION, dimension (100) :: X
        DOUBLE PRECISION, dimension (9) :: EXPO,ZNORM, CUM, GSC
C       DOUBLE PRECISION, dimension (400000) :: Z_tau_storage
C       INTEGER N,I,J,K,L,Nlya,NN,Nrun,nt, Iteration,Tau, Tau_index
        INTEGER N,I,J,K,L,Nlya,NN,Nrun,nt
        DOUBLE PRECISION :: A,B,C,D,E,F,R,S,Input
        DOUBLE PRECISION :: X_0
        DOUBLE PRECISION :: Rho,K0,M,Q
        DOUBLE PRECISION :: H
        COMMON/c1/N
        COMMON/c2/A,B,C,D,E,F,R,S,Input
        COMMON/c3/X_0
        COMMON/c4/Rho,K0,M,Q
C       COMMON/c5/z_tau,Tau

        A = 1.0D0
        B = 3.0D0
        C = 1.0D0
        D = 5.0D0
        E = 0.3D0
        F = 0.2D0
        R = 0.006D0
        S = 5.2D0
        Input = 0.8D0
        X_0 = -1.56D0
        Rho = 0.7D0
        K0 = 0.87D0
        M = 1.0D0
        Q = 0.005D0


        Open(7,File="Lya_k_rho_0.7_m_1.txt",Status='unknown')
        Open(8,File="Max_Lya_k_rho_0.7_m_1.txt",Status='unknown')
C
        N=5
        NN=(N*N)+N
C

        NRUN = 2000
        NLYA = 100*NRUN
        H = 0.01
c
        K0 = -1.30D0

        DO WHILE (K0 .LE. 0.60D0)
C         M  = 0.0 D0
C          DO WHILE (M .LE. 1.0D0)

C	INITIAL CONDITION FOR THE NONLINEAR SYSTEM
        X(1) = 0.0D0
        X(2) = 0.0D0
        X(3) = 0.0D0
        X(4) = 1.0D0
        X(5) = 0.0D0

!
C	INITIAL CONDITION FOR LINEAR SYSTEM
        DO I = N+1,NN
        X(I)=0.0D0

        END DO
C
        DO I=1,N
        X((N+1)*I)=1.0D0
        CUM(I)=0.0D0
        END DO
C
        DO I= 1,NLYA
        CALL RK2(X,H)
        END DO
C
        DO I = 1,NLYA
        CALL RK4(X,H)
C       WRITE(7,*)X(1),X(2),X(3),X(4),X(5)


C	NORMALIZE FIRST VECTOR
        ZNORM(1)=0.0D0
        DO J=1,N
        ZNORM(1)=ZNORM(1)+X(N*J+1)**2
        END DO
        ZNORM(1)=SQRT(ZNORM(1))
        DO J=1,N
        X(N*J+1)=X(N*J+1)/ZNORM(1)
        END DO
C
C	GENERATE THE NEW ORTHONORMAL SET OF VECTORS
        DO 40 J=2,N
C
C	GENERATE J-1 GSR COEFFICIENTS
        DO 10 K=1,J-1
        GSC(K)=0.0D0
        DO 10 L=1,N
        GSC(K)=GSC(K)+X(N*L+J)*X(N*L+K)
10      CONTINUE
C
C	CONSTRUCT A NEW VECTOR
        DO 20 K=1,N
        DO 20 L=1,J-1
        X(N*K+J)=X(N*K+J)-GSC(L)*X(N*K+L)
20      CONTINUE
C
C	CALCULATE THE VECTOR'S NORM
        ZNORM(J)=0.0D0
        DO 30 K=1,N
        ZNORM(J)=ZNORM(J)+X(N*K+J)**2
30      CONTINUE
        ZNORM(J)=SQRT(ZNORM(J))
C
C	NORMALIZE THE NEW VECTOR
        DO 40 K=1,N
        X(N*K+J)=X(N*K+J)/ZNORM(J)
40      CONTINUE
C
C	UPDATE RUNNING VECTOR MAGNITUDES
        DO K=1,N
        CUM(K)=CUM(K)+DLOG(ZNORM(K))/DLOG(2.0D0)
        END DO
C
        END DO
        DO K=1,N
        EXPO(K)=CUM(K)/(H*FLOAT(NLYA))
        END DO
        WRITE(7,*)K0, EXPO(1), EXPO(2), EXPO(3), EXPO(4), EXPO(5)
        WRITE(8,*)K0,MAX(EXPO(1),EXPO(2),EXPO(3),EXPO(4),EXPO(5)),M

C       M  = M + 0.005D0
C       END DO
        K0 = K0 + 0.004750D0
        END DO
        STOP
        END

C	****************************************************************
        SUBROUTINE RK4(X,H)
        IMPLICIT None
        INTEGER N,I,NN
        DOUBLE PRECISION :: H
        DOUBLE PRECISION, dimension (100) :: X,TEMP,AK1,AK2,AK3
        DOUBLE PRECISION, dimension (100) :: AK4,PRIME
        COMMON/c1/N
C
        NN=N*N+N

C
        DO I=1,NN
        TEMP(I)=X(I)
        END DO
C
        CALL DERIVE(TEMP,PRIME)
        DO I=1,NN
        AK1(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK1(I)/2.0D0
        END DO
C
        CALL DERIVE(TEMP,PRIME)
        DO I=1,NN
        AK2(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK2(I)/2.0D0
        END DO
C
        CALL DERIVE(TEMP,PRIME)
        DO I=1,NN
        AK3(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK3(I)
        END DO
C
        CALL DERIVE(TEMP,PRIME)
        DO I=1,NN
        AK4(I)=H*PRIME(I)
        X(I)=X(I)+1/6.0D0*(AK1(I)+2.0D0*(AK2(I)+AK3(I))+AK4(I))

        END DO
C
        RETURN
        END
C
C       **************************************************************
C
        SUBROUTINE DERIVE(X,PRIME)
        IMPLICIT None
        DOUBLE PRECISION, dimension (100):: X, PRIME
        INTEGER N,I
        DOUBLE PRECISION :: A,B,C,D,E,F,R,S,Input
        DOUBLE PRECISION :: X_0
        DOUBLE PRECISION :: Rho,K0,M,Q
        DOUBLE PRECISION :: H
        COMMON/c1/N
        COMMON/c2/A,B,C,D,E,F,R,S,Input
        COMMON/c3/X_0
        COMMON/c4/Rho,K0,M,Q
C       COMMON/c5/z_tau,Tau

        PRIME(1) = X(2)-(A * X(1)**3) + (B * X(1)**2)
     *  + K0*(E+ F*X(4)**2)*X(1) + Rho*X(5)*X(1) + Input
        PRIME(2) = C - (D * X(1) ** 2) - X(2)
        PRIME(3) = R * (S * (X(1) - X_0) - X(3))
        PRIME(4) = -X(4)+M*(ABS(X(4)+1.0D0)-ABS(X(4)-1.0D0))+X(1)
        PRIME(5) = X(1) - Q*X(5)
C
        DO I= 0,N-1
        PRIME(6+I) = (-3.0D0*A*X(1)**2 + 2.0d0*B*X(1)
     *  + K0*(E+F*X(4)**2) + Rho*X(5))*X(6+I) + X(11+I)
     *  + 2.0D0*K0*F*X(1)*X(4)*X(21+I) + Rho*X(1)*X(26+I)
        PRIME(11+I) = -2.0d0*D*X(1)*X(6+I) - X(11+I)
        PRIME(16+I) = R*S*X(6+I) - R*X(16+I)
        IF ( ABS(X(4)) .LT. 1) THEN
        PRIME(21+I) = X(6+I) + (2*M - 1)*X(21+I)
        ELSE
        PRIME(21+I) = X(6+I) - X(21+I)
        END IF
        PRIME(26+I) = X(6+I) - Q*X(26+I)
	END DO
C
        RETURN
        END
C
C	****************************************************************
        SUBROUTINE RK2(X,H)
        IMPLICIT None
        INTEGER N,I
        DOUBLE PRECISION :: H
        DOUBLE PRECISION, dimension (100) :: X,TEMP,AK1,AK2,AK3
        DOUBLE PRECISION, dimension (100) :: AK4,PRIME
        COMMON/c1/N
C
        DO I=1,N
        TEMP(I)=X(I)
        END DO
C
        CALL DER(TEMP,PRIME)
        DO I=1,N
        AK1(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK1(I)/2.0D0
        END DO
C
        CALL DER(TEMP,PRIME)
        DO I=1,N
        AK2(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK2(I)/2.0D0
        END DO
C
        CALL DER(TEMP,PRIME)
        DO I=1,N
        AK3(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK3(I)
        END DO
C
        CALL DER(TEMP,PRIME)
        DO I=1,N
        AK4(I)=H*PRIME(I)
        X(I)=X(I)+1/6.0D0*(AK1(I)+2.0D0*(AK2(I)+AK3(I))+AK4(I))
        END DO
        RETURN
        END
C
C       **************************************************************
C
        SUBROUTINE DER(X,PRIME)
        IMPLICIT None
        DOUBLE PRECISION, dimension (100):: X, PRIME
        INTEGER N,I
        DOUBLE PRECISION :: A,B,C,D,E,F,R,S,Input
        DOUBLE PRECISION :: X_0
        DOUBLE PRECISION :: Rho,K0,M,Q
        DOUBLE PRECISION :: H
        COMMON/c1/N
        COMMON/c2/A,B,C,D,E,F,R,S,Input
        COMMON/c3/X_0
        COMMON/c4/Rho,K0,M,Q
C       COMMON/c5/z_tau,Tau

        PRIME(1) = X(2)-(A * X(1)**3) + (B * X(1)**2)
     *  + K0*(E+ F*X(4)**2)*X(1) + Rho*X(5)*X(1) + Input
        PRIME(2) = C - (D * X(1) ** 2) - X(2)
        PRIME(3) = R * (S * (X(1) - X_0) - X(3))
        PRIME(4) = -X(4)+M*(ABS(X(4)+1.0D0)-ABS(X(4)-1.0D0))+X(1)
        PRIME(5) = X(1) - Q*X(5)

        RETURN
        END
C

