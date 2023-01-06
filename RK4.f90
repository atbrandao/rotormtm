program ORB_PC

real dt_base, ti, tf, E_max, omg, Per, e, delta, k_c, dd, mr, ms, cpt_i, cpt_f, dt_min   !Parametros reais
real*8 dt, gg, aux, th1, th, pi
real*8, dimension(8) :: u0, Bg, BB, CNp                                        
real, allocatable :: U(:,:), tt(:), u_pc1(:,:), u_pc2(:,:)  !Variaveis de armazenamento final de dados
real*8, dimension(3) :: auxx1
real*8, dimension(2) :: auxx2
integer, dimension(20) :: cont
real*8, dimension(2) :: t
real*8, dimension(8,2) :: uu
real*8, dimension(4) :: yy
real, dimension(8,8) :: A                !Matriz A do espaço de estados
integer, dimension(4,8) :: C             !Matriz C do espaço de estados
integer i, k, kk, j, N, impact, q, np, period       !Contadores inteiros
real gr                                  !Gerais

call CPU_TIME(cpt_i)

open(3,file = 'entrada.dat')

read(3,'(////////////,25X,F3.1)') ti
read(3,'(25X,F5.1)') tf
read(3,'(25X,E7.1)') E_max
read(3,'(25X,F7.5)') dt_base
read(3,'(25X,I3)') np        !A cada período de excitação serão armazenados 'np' posições calculadas
read(3,'(25X,F5.3)') delta
read(3,'(25X,E5.1)') k_c
read(3,'(25X,F6.4)') e
read(3,'(25X,F6.1)') dd
read(3,'(25X,I5)') period

close(3)

pi=3.141592653589793
print *, k_c
call DEFSYS(A,C,omg,mr,ms) !Chamando subrotina para definir as matrizes do sistema

!if (k_c .eq. 1e8) then
	!omg=2*pi*omg
!end if

Per = 2 * 3.14 / omg

N=ceiling(1.1*np*tf/Per)
print *, np, tf, Per/np, N, ti
print *, '---'
allocate(U(8,N),tt(N), u_pc1(3,N/np),u_pc2(2,N/np))



gr = 9.81
BB(:)=0
CNp(:)=0
Bg(:)=0
Bg(4)=-gr
Bg(8)=-gr

U(:,:)=0
tt(:)=0
u_pc1(:,:)=0
u_pc2(:,:)=0
cont(:)=0
yy(:)=0

u0 = (/ 0,	0, 0,0	,0, 0,	0,	0 /) 


uu(:,1) = u0(:)
uu(:,2) = u0(:)
U(:,1) = u0(:)

yy = matmul(C,uu(:,1))
gg = sqrt((yy(1)-yy(3))**2+(yy(2)-yy(4))**2)-delta

t(:) = ti
tt(1) = ti


if (gg .ge. 0) then
    impact = 1
else
    impact = 0
end if

k = 2
dt = dt_base
cont = 0

kk = 1
q = 1
i = 0
j=0;
dt_min=dt_base
!!!!!!!!!!!!!!!!!!!!
open(4,file = 'saida.txt')

write(4,*), 'Tempo Gap Xr Vxr Yr Vyr Xs Vxs Ys Vys'

do while (tt(kk) .lt. tf)
    t(1)=tt(kk)
    uu(:,1)=U(:,kk)
    !print *, U(:,kk)
    !print *, uu(:,2)
    kk=kk+1
    !print *, t(k)
do while (t(2)-t(1) .lt. Per/np)
	j=j+1;
    uu(:,1)=uu(:,2)
    !print *, uu(:,2)
    yy=matmul(C,uu(:,1))
    gg=sqrt((yy(1)-yy(3))**2+(yy(2)-yy(4))**2)-delta
    
    if (gg .ge. 0) then
        
        impact=1      
    else                
        impact=0
        if (gg .le. -dd) then
            dt=dt_base
        end if
    end if
    aux=gg+delta
    
    if (impact .eq. 1) then
		CNp(:) = 0
        CNp(2) = (yy(3)-yy(1))/aux/mr 
        CNp(4) = (yy(4)-yy(2))/aux/mr 
        CNp(6) = (yy(1)-yy(3))/aux/ms
        CNp(8) = (yy(2)-yy(4))/aux/ms 
        
    else
        CNp(:) = (/ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 /)
    end if
    
    
    
    BB=Bg+(CNp*k_c*gg) 
    
    
    
    !print *, impact
    !print *, t(k)
    !print *, RK4(uu(:,1),A,BB,dt,t(k))
    
    
    uu(:,2)=RK4(uu(:,1),BB,dt,t(2)) 
    
    
    yy=matmul(C,uu(:,2))
    
    gg=sqrt((yy(1)-yy(3))**2+(yy(2)-yy(4))**2)-delta    
    
    
    if ((gg .ge. 0) .and. (impact .eq. 0)) then !% | gp<0 & impact | erroi% | gp<0 & impact  
		if (i .lt. 20) then
			i=i+1
		end if
        cont(i)=0;
        aux=Erro(dt)
        do while (Erro(dt) .gt. E_max) !% | Err<E_min
                    
            dt=dt*(E_max/Erro(dt))**(1)
                        
            uu(:,2)=RK4(uu(:,1),BB,dt,t(2))
            
            yy=matmul(C,uu(:,2))
            
            gg=sqrt((yy(1)-yy(3))**2+(yy(2)-yy(4))**2)-delta
            cont(i)=cont(i)+1
            !print *, 'erro'
        end do
        !print *, 'contato', t(k), '-', cont(i), '-', aux
        if (dt .lt. dt_min) then
			dt_min=dt
			print *, 'dt_min= ', dt
		end if
    
    end if   
    
    
    !if ((sin(omg*t(k)) .lt. 0) .and. (sin(omg*(t(k)+dt)) .gt. 0)) then
    if  (( sin(omg*t(2)) .gt. (sin(omg*(t(2)+dt))) ) .and. ( sin(omg*(t(2)-dt)) .lt. sin(omg*t(2))) ) then
		
		aux=2*pi
		th=2*pi*resto(omg*t(2),aux)
        th1=2*pi*resto(omg*(t(2)+dt),aux)
        auxx1=(uu(5:7,2)-uu(5:7,1))*(pi/2-th1)/(th1-th)
        auxx2=(uu(1:2,2)-uu(1:2,1))*(pi/2-th1)/(th1-th)
        
        u_pc1(:,q)=uu(5:7,2)+auxx1
        u_pc2(:,q)=uu(1:2,2)+auxx2
        q=q+1
        
    end if
    
    
    if ( nint(100*(t(2)-ti)/(tf-ti)) .gt. nint(100*(t(2)-dt-ti)/(tf-ti)) ) then
		!call system('cls')
		print *, nint(100*(t(2)-ti)/(tf-ti))
	end if
	
    t(2)=t(2)+dt
	
end do
U(:,kk)=uu(:,2)  !Armazenando ultima posição calculada nas variaveis finais
tt(kk)=t(2)

write(4,*)
write(4,"(3f18.13)",advance='no'), tt(kk), -gg
write(4,"(a1)",advance='no'), ' '
do i=1,8
	write(4,"(3f13.10)",advance='no'), U(8,kk)
	write(4,"(a1)",advance='no'), ' '
end do

end do
!!!!!!!!!!!!!!!!!!
print *, '---', j
close(4)

open(5,file='orbitas.txt')
write(5,*), 'Xr Vxr Xs Ys Vxs PCXr PCVXr PCXs PCVxs PCYs'

k=kk-np*period
 j=1
do i=k,kk
	if (j .le. period) then
		write(5,*)
		write(5,"(3f18.13)",advance='no'), U(1,i)
		write(5,"(a1)",advance='no'), ' '
		write(5,"(3f18.13)",advance='no'), U(2,i)
		write(5,"(a1)",advance='no'), ' '
		write(5,"(3f18.13)",advance='no'), U(5,i)
		write(5,"(a1)",advance='no'), ' '
		write(5,"(3f18.13)",advance='no'), U(7,i)
		write(5,"(a1)",advance='no'), ' '
		write(5,"(3f18.13)",advance='no'), U(6,i)
		write(5,"(a1)",advance='no'), ' '
		write(5,"(3f18.13)",advance='no'), u_pc2(:,q-j)
		write(5,"(a1)",advance='no'), ' '
		write(5,"(3f18.13)",advance='no'), u_pc1(:,q-j)
		write(5,"(a1)",advance='no'), ' '
	else
		write(5,*), U(1:2,i), U(5,i), U(7,i), U(6,i)
	end if
	j=j+1
end do

close(5)

call CPU_TIME(cpt_f)

print *, 'Tempo decorrido: ', cpt_f-cpt_i

contains

function RK4(u1, BB1, dt1, t1)   !Função Runge-Kutta 4ª Ordem
real*8, dimension(8) :: RK4, u1, k1, k2, k3, k4, BB1, auxx
real*8 dt1, t1

auxx(:)=0
auxx(2)=e*omg**2*cos(omg*t1)
auxx(4)=e*omg**2*sin(omg*t1)
auxx=auxx+BB1

k1=dt*(matmul(A,u1) + auxx )

auxx(:)=0
auxx(2)=e*omg**2*cos(omg*(t1+dt1/2))
auxx(4)=e*omg**2*sin(omg*(t1+dt1/2))
auxx=auxx+BB1

k2=dt*(matmul(A,(u1+k1/2))+auxx)
k3=dt*(matmul(A,(u1+k2/2))+auxx)

auxx(:)=0
auxx(2)=e*omg**2*cos(omg*(t1+dt1))
auxx(4)=e*omg**2*sin(omg*(t1+dt1))
auxx=auxx+BB1

k4=dt*(matmul(A,(u1+k3))+auxx)

RK4=u1+(k1+2*k2+2*k3+k4)/6

end function RK4


function Erro(dt)  !Função Erro
real*8 Erro, dt
real*8, dimension(8) :: uu2,Dirct2,BB2
real*8, dimension(4) :: yy2,CNp2
real*8 gg2, aaux
integer j

uu2=RK4(uu(:,1),BB,dt/2,t(k))
yy2=matmul(C,uu2)

gg2=sqrt((yy2(1)-yy2(3))**2+(yy2(2)-yy2(4))**2)-delta

   aaux=gg2+delta;
	if (gg2 .ge. 0) then
		CNp2=(/ (yy2(3)-yy2(1))/aaux/mr,(yy2(4)-yy2(2))/aaux/mr,-(yy2(3)-yy2(1))/aaux/ms,-(yy2(4)-yy2(2))/aaux/ms /)
	else
		CNp2=(/0,0,0,0/)
	end if
   
   
   Dirct2=(/0,0,0,0,0,0,0,0/)
   do j=1,4
       Dirct2(2*j)=CNp2(j)
   end do
   
  
   
   BB2=Bg+Dirct2(:)*k_c*gg2


uu2=RK4(uu2,BB2,dt/2,t(k)+dt/2);

yy2=matmul(C,uu2)
Erro=max(abs(yy(1)-yy2(1)),abs(yy(2)-yy2(2)),abs(yy(3)-yy2(3)),abs(yy(4)-yy2(4)))/15;

end function Erro

function resto(a,b)
real*8 a,b,resto

resto=a/b-int(a/b)

end function resto



end program 

subroutine DEFSYS(A_sr,C_sr,omg_sr,mr,ms)

! Subrotina que lê os parâmetros do sistema q gera as matrizes do espaço de estados

real mr, ms, kr, ks, cr, cs, ws, wr, csi_r, csi_s, omg_sr        !Parametros do sistema
real, dimension(4,4) :: MM, KK, CC          !Parametros da EDO de segundo grau
real, dimension(8,8) :: A_sr        !Matriz A do espaço de estados
integer, dimension(4,8) :: C_sr     !Matriz C do espaço de estados
integer i

open(3,file = 'entrada.dat')

read(3,'(///,25X,F4.0)') ws
read(3,'(25X,F4.0)') wr
read(3,'(25X,F3.0)') ms
read(3,'(25X,F3.0)') mr
read(3,'(25X,F4.2)') csi_r
read(3,'(25X,F4.2)') csi_s
read(3,'(25X,F8.4)') omg_sr

close(3)

kr = wr*wr*mr
ks = ws*ws*ms
cr = csi_r*2*sqrt(kr*mr)
cs = csi_s*2*sqrt(ks*ms)

KK(:,:) = 0
MM(:,:) = 0
CC(:,:) = 0
A_sr(:,:) = 0

do i = 1,2
	KK(i,i) = kr
	MM(i,i) = mr
	CC(i,i) = cr
end do

do i = 3,4
	KK(i,i) = ks
	MM(i,i) = ms
	CC(i,i) = cs
end do

do i = 1,8
    if (i .eq. 2*int(i/2)) then
        do j=1, 7, 2
            A_sr(i,j)=-KK(i/2,(j+1)/2)/MM(i/2,i/2)
        end do
        do j=2, 8, 2
            A_sr(i,j)=-CC(i/2,(j)/2)/MM(i/2,i/2)
        end do
    else
        A_sr(i,:)=0
        A_sr(i,i+1)=1
    end if    
end do

C_sr(:,:)=0
do i=1, 4
    C_sr(i,2*i-1)=1;
end do

return
end





