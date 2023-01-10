

    program rk4_fortran
    
    ! implicit none
    
    integer ndof, n_cpar, i, j, k, n_f, n_probe
    real*8 alpha, beta, dummy, dt, tf, omg
    real*8, allocatable :: A(:,:), Minv(:,:), Snl(:,:), Snl_stsp(:,:), f(:,:), x0(:,:) ! Input variables
    real*8, allocatable :: x_out(:,:), x_aux(:,:), B(:,:), t(:)
    integer, allocatable :: probe_dof(:)
    real*8, dimension(3) :: f_aux

    open(3,file = 'data.dat')
    read(3,*)
    read(3,*) n_cpar
    read(3,*) n_f
    allocate(f(n_f, 3))
    do i = 1, n_f
        read(3,*) f_aux
        !do j = 1, 3
        f(i, :) = f_aux(:)
        !end do
    end do
    read(3,*) dt
    read(3,*) tf
    read(3,*) omg    
    read(3,*) ndof
    allocate(x0(2 * ndof, 1))
    read(3,*) x0
    read(3,*) n_probe
    allocate(probe_dof(n_probe))
    read(3,*) probe_dof
    
    read(3,*)
    read(3,*) ndof
    allocate(A(2 * ndof, 2 * ndof))
    allocate(Minv(2 * ndof, 2 * ndof))
    allocate(Snl(ndof, ndof))
    allocate(Snl_stsp(2 * ndof, 2 * ndof))

    read(3,*) alpha
    read(3,*) beta
    read(3,*) A
    A = transpose(A)
    read(3,*) Minv
    Minv = transpose(Minv)
    read(3,*) Snl
    Snl = transpose(Snl)
    
    do i=1, ndof
        do j=1, ndof
            Snl_stsp(i, j) = 0
            Snl_stsp(i, j + ndof) = 0
            Snl_stsp(i + ndof, j + ndof) = 0
            Snl_stsp(i + ndof, j) = - Snl(i, j)
        end do
    end do

    close(3)
    
    N = floor(tf / dt) + 1
    allocate(t(N))
    allocate(B(2 * ndof, 3))
    allocate(x_out(n_probe, N))
    allocate(x_aux(2 * ndof, 1))
    
    t(1) = 0
    
    x_aux = x0
    do j = 1, n_probe
        x_out(j, 1) = x_aux(probe_dof(j) + 1, 1)
    end do
    
    do i = 2, N
        t(i) = t(i-1) + dt
        
        B(:,:) = 0
        do j = 1, n_f
            k = nint(f(j, 1)) + ndof
            B(k, 1) = f(j, 2) * cos(omg * t(i-1)) + f(j, 3) * sin(omg * t(i-1))
            B(k, 2) = f(j, 2) * cos(omg * t(i-1) + dt/2) + f(j, 3) * sin(omg * t(i-1) + dt/2)
            B(k, 3) = f(j, 2) * cos(omg * t(i-1) + dt) + f(j, 3) * sin(omg * t(i-1) + dt)
        end do
            
        B = matmul(Minv, B)
        
        x_aux = RK4(B, x_aux, dt)
        do j = 1, n_probe
            x_out(j, i) = x_aux(probe_dof(j) + 1, 1)
        end do
        print *, t(i)
    end do
    

    open(4, file = 'saida.txt')
    
    do i = 1, N
        do j = 1, n_probe
            
            write(4, '(f18.13)', advance='no'), x_out(j, i)
            
        end do
        write(4, *), ''
    end do
    
    close(4)
    
    
    
    
    
    contains
    
    function RK4(B, x, dt)   !Função Runge-Kutta 4ª Ordem
    
        real*8, dimension(2 * ndof, 1) :: RK4, x, k1, k2, k3, k4, x1, x2, x3, b_aux
        real*8, dimension(2 * ndof, 3) :: B
        real*8 dt
        
        b_aux(:, 1) = B(:, 1)
        k1 = matmul(A, x) + b_aux + matmul(Minv, (beta * matmul(Snl_stsp, x) + alpha * matmul(Snl_stsp, x) ** 3))
        
        x1 = x + k1 * dt / 2
        b_aux(:, 1) = B(:, 2)
        k2 = matmul(A, x1) + b_aux + matmul(Minv, (beta * matmul(Snl_stsp, x1) + alpha * matmul(Snl_stsp, x1) ** 3))
        
        x2 = x + k2 * dt / 2
        k3 = matmul(A, x2) + b_aux + matmul(Minv, (beta * matmul(Snl_stsp, x2) + alpha * matmul(Snl_stsp, x2) ** 3))
        
        x3 = x + k3 * dt
        b_aux(:, 1) = B(:, 3)
        k4 = matmul(A, x3) + b_aux + matmul(Minv, (beta * matmul(Snl_stsp, x3) + alpha * matmul(Snl_stsp, x3) ** 3))

        RK4 = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    

    end function RK4





end program rk4_fortran

