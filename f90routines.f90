MODULE f90routines

      IMPLICIT NONE

      CONTAINS
      SUBROUTINE f90sparsecentrality(IC, AA, JA, M, N, NNZ, centrality_out)
                INTEGER, INTENT(in)                     :: M, N, NNZ
                INTEGER, DIMENSION(N),   INTENT(in)     :: IC
                INTEGER, DIMENSION(NNZ), INTENT(in)     :: AA, JA

                INTEGER, DIMENSION(M),   INTENT(out)    :: centrality_out
                !f2py depend(N) site, visited, x
                INTEGER, DIMENSION(M)                   :: site, visited, x

                INTEGER :: i, j, k, l, k1, k2

                centrality_out = 0

                !$OMP PARALLEL DO PRIVATE(site, visited, x, l, k1, k2, k)
                DO i = 1,M
                       site = 0
                       site(i) = 1
                       visited = -1
                       WHERE (site /= 0) visited = 0
                       DO WHILE (COUNT(visited == 0) > 0)
                               centrality_out(i) = centrality_out(i) + 1
                               WHERE (visited == 0) visited = 1
                               x = site
                               DO l = 1, M
                                       k1 = IC(l)
                                       k2 = IC(l + 1) - 1
                                       site(l) = 0
                                       DO k = k1, k2
                                               site(l) = site(l) + AA(k)*x(JA(k))
                                       END DO
                               END DO
                               WHERE (site > 0 .AND. visited == -1) visited = 0
                       END DO
                END DO
                !$OMP END PARALLEL DO

      END SUBROUTINE f90sparsecentrality

      SUBROUTINE f90centrality(a, N, centrality_out)

              INTEGER, INTENT(in) :: N
              INTEGER, DIMENSION(N,N), INTENT(in) :: a

              INTEGER, DIMENSION(N) :: site, visited
              !f2py depend(N) site, visited
              INTEGER :: i

              INTEGER, DIMENSION(N),INTENT(out) :: centrality_out

              centrality_out = 0

            ! the algorithm cycles through each node
            ! then applies the adjacency matrix to
            ! the notes site until no new nodes can
            ! be reached. The number of applications
            ! is equal to the longest shortest path
            ! through that node.
            !
            ! visited is equal to
            ! -1 if node is unvisited
            ! 0  if node is newly visited
            ! 1  if node was previously visited
            
            !!$OMP PARALLEL DO PRIVATE(site, visited)
              DO i = 1,N
                     site = 0
                     site(i) = 1
                     visited = -1
                     WHERE (site /= 0) visited = 0
                     DO WHILE (COUNT(visited == 0) > 0)
                             centrality_out(i) = centrality_out(i) + 1
                             WHERE (visited == 0) visited = 1
                             site = MATMUL(site,a)
                             WHERE (site > 0 .AND. visited == -1) visited = 0
                     END DO
             END DO
           !!$OMP END PARALLEL DO

      END SUBROUTINE f90centrality

      !SUBROUTINE f90waterflow(N_land, land_list, map_height, map_width, precipitation, elevation, flow, water)
      SUBROUTINE f90waterflow(land_list, elevation, precipitation, neighbours, max_x, &
                      max_y, N_land, err, flow, water)

              ! N_land: number of land cells
              ! land_list: coordinates of land cells (x,y)
              ! map_hight, map_width: map dimensions in number of cells
              ! precipitation: square map with precipitation value for each cell
              ! 

              INTEGER, INTENT(in)                               :: N_land
              INTEGER, DIMENSION(2,N_land), INTENT(in)          :: land_list

              INTEGER, INTENT(in)                               :: max_x, max_y
              REAL, DIMENSION(max_x,max_y), INTENT(in)          :: precipitation, elevation
              REAL, DIMENSION(max_x,max_y), INTENT(out)         :: flow, water
              INTEGER, DIMENSION(2,9), INTENT(in)               :: neighbours
              INTEGER, INTENT(out)                              :: err

              REAL, DIMENSION(max_x,max_y)                      :: Z
              REAL, DIMENSION(N_land)                           :: drop_volume
              INTEGER, DIMENSION(2,N_land)                      :: drop_coordinates
              !f2py depend(N_land) drop_volume, drop_coordinates
              !f2py depend(map_height,map_width) Z
              REAL                                              :: water_elev_new, water_elev_old
              INTEGER                                           :: i, j, k, x, y, x_tmp, y_tmp, x_new, y_new, step

              err = max_x

              !set flow to zero at begin of calculation
              flow = 0
              !set water to drop_volumes on cells
              water = precipitation

              !Initialize drop coordinates and drop volumes in a list
              DO i = 1,N_land
                !array indices start with 1 in fortran and 0 in python!!
                drop_coordinates(1,i) = land_list(1,i) + 1
                drop_coordinates(2,i) = land_list(2,i) + 1
                drop_volume(i) = precipitation(land_list(1,i)+1,land_list(2,i)+1)
              END DO
              
              DO step = 1,10
                !Build water level map from elevation + drop volume
                Z = water + elevation
                water = 0
                !$OMP PARALLEL DO REDUCTION(+:water,flow) PRIVATE(x,y,water_elev_new,water_elev_old,k,j,x_tmp,y_tmp,x_new,y_new)
                DO i = 1,N_land
                  !iterate over all drops with coordinates (x,y) (one per land cell)
                  x = drop_coordinates(1,i)
                  y = drop_coordinates(2,i)
                  water_elev_new = 0
                  water_elev_old = 10000
                  k = 0
                  DO j = 1,9
                    !find cell (x_new,y_new) with lowest water level in neighbourhood of drop
                    !make sure that indices stay whithin bounds!!
                    x_tmp = MOD(x + neighbours(1,j)-1,max_x)+1
                    y_tmp = MOD(y + neighbours(2,j)-1,max_y)+1
                    !IF(x_tmp .NE. x + neighbours(1,j) .OR. y_tmp .NE. y + neighbours(2,j)) THEN
                    !        WRITE(*,*) 'outbound', max_x, max_y
                    !        WRITE(*,*) x_tmp, x + neighbours(1,j)
                    !        WRITE(*,*) y_tmp, y + neighbours(2,j)
                    !ENDIF
                    !IF(x_tmp == 0 .OR. y_tmp == 0) THEN
                    !        WRITE(*,*) 'ZEROOOO!!!'
                    !ENDIF
                    water_elev_new = Z(x_tmp,y_tmp)
                    IF (water_elev_new < water_elev_old) THEN
                        x_new = x_tmp
                        y_new = y_tmp
                        water_elev_old = water_elev_new
                        k = j
                    ENDIF
                  END DO
                  !move drop to cell with lovest water level
                  drop_coordinates(1,i) = x_new
                  drop_coordinates(2,i) = y_new
                  !if drop moved, add volume to flow of cell
                  IF(k .NE. 5) THEN
                    flow(x,y) = flow(x,y) + drop_volume(i)
                  ENDIF
                  !add volume to water of cell
                  water(x,y) = water(x,y) + drop_volume(i)
                  !move drop volume on water level map Z
!                  Z(x,y) = Z(x,y) - drop_volume(i)
!                  Z(x_new,y_new) = Z(x_new,y_new) + drop_volume(i)
                END DO
                !$OMP END PARALLEL DO
              END DO

      END SUBROUTINE f90waterflow

END MODULE f90routines


