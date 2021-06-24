#include <math.h>
#include "mpi.h"
#include<iostream>
#include<cstdlib>
#include"EasyBMP.h"

using namespace std;

#define PI 3.14159
#define T_LOW 60
#define T_HIGH 85
#define NXPROB 512
#define NYPROB 512
#define MAX_SIZE 3

int getOrientation(float);

bool isBetween(float, float, float, float, float);

bool isFirstMax(int a, int b, int c);

unsigned int ROWS;
unsigned int COLUMNS;
char DEPTH;
///---------------------------------get size of image-----------------------------------
void printFileInfo(BMP image) {
    cout << endl << "File info:" << endl;
    cout << image.TellWidth() << " x " << image.TellHeight()
         << " at " << image.TellBitDepth() << " bpp" << endl << endl;
}
///-----------------------get convolution value from image*mask-------------------------
float convolve(int con[][MAX_SIZE], int dim, float divisor, int i, int j, int *imArray) {
    int midx = dim / 2;//1
    int midy = dim / 2;//1
    float weightedSum = 0;
    for (int x = i - midx; x < i + dim - midx; x++) {
        for (int y = j - midy; y < j + dim - midy; y++) {
            weightedSum += divisor * (float) (con[x - i + midx][y - j + midy]
                                              * imArray[x * NXPROB + y]);
        }
    }
    return weightedSum;
}
///---------------------select approximate direction value-------------------
int getOrientation(float angle) {
    if (isBetween(angle, -22.5, 22.5, -180, -157.5) || isBetween(angle, 157.5, 180, -22.5, 0))
        return 0;
    if (isBetween(angle, 22.5, 67.5, -157.5, -112.5))
        return 45;
    if (isBetween(angle, 67.5, 112.5, -112.5, -67.5))
        return 90;
    if (isBetween(angle, 112.5, 157.5, -67.5, -22.5))
        return 135;

    return -1;
}
///---------------------select if a<arg<b or c<arg<d-------------------------
bool isBetween(float arg, float a, float b, float c, float d) {
    if ((arg >= a && arg <= b) || (arg >= c && arg <= d)) {
        return true;
    } else {
        return false;
    }
}
///---------------------select if a is bigger than b, c-------------------------
bool isFirstMax(int a, int b, int c) {
    if (a > b && a > c) {
        return true;
    }
    return 0;
}

///--------------------------------main----------------------------------------
int main(int argc, char **argv) {
    int rank, size, i, j;
    int i_first, i_last;
    double T1, T2;
    MPI_Status status;
    int *imageArray;
    int xlocal[2][(NXPROB / 4) + 2][NYPROB];
    float *thetaslocal;
    int *magArraylocal;
    BMP InputIMG;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        if (!InputIMG.ReadFromFile("./lena512.bmp")) {
            cout << "Invalid File Name..." << endl;
            return EXIT_FAILURE;
        }

        printFileInfo(InputIMG);
        COLUMNS = InputIMG.TellWidth();  // num cols
        ROWS = InputIMG.TellHeight(); // num rows
        DEPTH = InputIMG.TellBitDepth();
        ///--------------------------------memory allocate-------------------------------
        imageArray = new int[ROWS * COLUMNS]; // row memory allocation
        ///----------------------------data(image) load------------------------------------
        int Temp;
        cout << "Saving Brightness values" << endl;
        for (int j = 0; j < ROWS; j++) {
            for (int i = 0; i < COLUMNS; i++) {
                Temp = (int) floor(0.299 * InputIMG(i, j)->Red +
                                   0.587 * InputIMG(i, j)->Green +
                                   0.114 * InputIMG(i, j)->Blue);
                imageArray[i + j * COLUMNS] = Temp;
            }
        }
    }
    T1 = MPI_Wtime(); /* parallel start time */
    MPI_Scatter(&imageArray[0], NXPROB * (NXPROB / size), MPI_INT,
                xlocal[0][1], NXPROB * (NXPROB / size), MPI_INT,
                0, MPI_COMM_WORLD);
    ///-------data location(4) : task0-2~128  task1-1~128  task2-1~128  task3-1~127---------
    i_first = 1;//1
    i_last = NXPROB / size;//128
    if (rank == 0) i_first++;//2
    if (rank == size - 1) i_last--;//127
    ///-----------------------------image value exchange-----------------------------------
    /* Send up unless I'm at the top, then receive from below */
    if (rank < size - 1)
        MPI_Send(xlocal[0][NXPROB / size], NXPROB, MPI_INT, rank + 1, 0,
                 MPI_COMM_WORLD);//012 send to 123
    if (rank > 0)
        MPI_Recv(xlocal[0][0], NXPROB, MPI_INT, rank - 1, 0,
                 MPI_COMM_WORLD, &status);//recv from 012
    /* Send down unless I'm at the bottom */
    if (rank > 0)
        MPI_Send(xlocal[0][1], NXPROB, MPI_INT, rank - 1, 1,
                 MPI_COMM_WORLD);//123 send to 012
    if (rank < size - 1)
        MPI_Recv(xlocal[0][NXPROB / size + 1], NXPROB, MPI_INT, rank + 1, 1,
                 MPI_COMM_WORLD, &status);//recv from 123
    ///-----------------------------actual image processing------------------------------
    ///-----------------------------------gaussian---------------------------------------
    int gaussArray[3][3] = {{1, 2, 1},
                            {2, 4, 2},
                            {1, 2, 1}};
    float gaussDivisor = 1.0 / 16.0;
    for (i = 0; i < NXPROB / size + 2; i++)
        for (j = 0; j < NXPROB; j++)
            xlocal[1][i][j] = 0;
    float sum = 0.0;
    int dim = 3;
    for (int i = i_first; i <= i_last; i++) {
        for (int j = 1; j <= NXPROB - 1; j++) {
            sum = convolve(gaussArray, dim, gaussDivisor, i, j, xlocal[0][0]);
            xlocal[1][i][j] = (int) sum; //gaussian value
        }
    }
    ///-----------------------gather and scatter for noise cancel------------------------
    MPI_Gather(xlocal[1][1], NXPROB * (NXPROB / size), MPI_INT,
               &imageArray[0], NXPROB * (NXPROB / size), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(&imageArray[0], NXPROB * (NXPROB / size), MPI_INT,
                xlocal[0][1], NXPROB * (NXPROB / size), MPI_INT,
                0, MPI_COMM_WORLD);

    ///-------------------------------------sobel----------------------------------------
    float G_x, G_y, G;
    int sobel_y[3][3] = {{-1, 0, 1},
                         {-2, 0, 2},
                         {-1, 0, 1}};

    int sobel_x[3][3] = {{1,  2,  1},
                         {0,  0,  0},
                         {-1, -2, -1}};
    thetaslocal = new float[(NXPROB / size + 2) * NXPROB];
    magArraylocal = new int[(NXPROB / size + 2) * NXPROB];
    for (int i = i_first; i <= i_last; i++) {
        for (int j = 1; j <= NXPROB - 1; j++) {
            G_x = convolve(sobel_x, dim, 1, i, j, xlocal[0][0]);
            G_y = convolve(sobel_y, dim, 1, i, j, xlocal[0][0]);
            G = sqrt(G_x * G_x + G_y * G_y);
            thetaslocal[i * NXPROB + j] = getOrientation(180.0 * atan2(G_y, G_x) / PI);
            magArraylocal[i * NXPROB + j] = G;
        }
    }
    ///--------------------------theta and mag value exchange-----------------------------------
    /* Send up unless I'm at the top, then receive from below */
    if (rank < size - 1)
        MPI_Send(&thetaslocal[NXPROB / size], NXPROB, MPI_INT, rank + 1, 0,
                 MPI_COMM_WORLD);//012 send to 123
    if (rank > 0)
        MPI_Recv(&thetaslocal[0], NXPROB, MPI_INT, rank - 1, 0,
                 MPI_COMM_WORLD, &status);//recv from 012
    /* Send down unless I'm at the bottom */
    if (rank > 0)
        MPI_Send(&thetaslocal[1], NXPROB, MPI_INT, rank - 1, 1,
                 MPI_COMM_WORLD);//123 send to 012
    if (rank < size - 1)
        MPI_Recv(&thetaslocal[NXPROB / size + 1], NXPROB, MPI_INT, rank + 1, 1,
                 MPI_COMM_WORLD, &status);//recv from 123
    /* Send up unless I'm at the top, then receive from below */
    if (rank < size - 1)
        MPI_Send(&magArraylocal[NXPROB / size], NXPROB, MPI_INT, rank + 1, 0,
                 MPI_COMM_WORLD);//012 send to 123
    if (rank > 0)
        MPI_Recv(&magArraylocal[0], NXPROB, MPI_INT, rank - 1, 0,
                 MPI_COMM_WORLD, &status);//recv from 012
    /* Send down unless I'm at the bottom */
    if (rank > 0)
        MPI_Send(&magArraylocal[1], NXPROB, MPI_INT, rank - 1, 1,
                 MPI_COMM_WORLD);//123 send to 012
    if (rank < size - 1)
        MPI_Recv(&magArraylocal[NXPROB / size + 1], NXPROB, MPI_INT, rank + 1, 1,
                 MPI_COMM_WORLD, &status);//recv from 123
    ///------------------------------------nomax-----------------------------------------
    int theta = 0;
    for (int i = i_first; i <= i_last; i++) {
        for (int j = 1; j <= NXPROB - 1; j++) {
            theta = (int) thetaslocal[i * NXPROB + j];
            switch (theta) {
                case 0:
                    if (isFirstMax(magArraylocal[i * NXPROB + j], magArraylocal[(i + 1) * NXPROB + j],
                                   magArraylocal[(i - 1) * NXPROB + j])) {
                        xlocal[0][i][j] = magArraylocal[i * NXPROB + j]; // black
                    }else{
                        xlocal[0][i][j]=0;
                    }
                    break;

                case 45:
                    if (isFirstMax(magArraylocal[i * NXPROB + j], magArraylocal[(i + 1) * NXPROB + j + 1],
                                   magArraylocal[(i - 1) * NXPROB + j - 1])) {
                        xlocal[0][i][j] = magArraylocal[i * NXPROB + j]; // black
                    }else{
                        xlocal[0][i][j]=0;
                    }

                    break;

                case 90:
                    if (isFirstMax(magArraylocal[i * NXPROB + j], magArraylocal[i * NXPROB + j + 1],
                                   magArraylocal[i * NXPROB + j - 1])) {
                        xlocal[0][i][j] = magArraylocal[i * NXPROB + j]; // black
                    }else{
                        xlocal[0][i][j]=0;
                    }
                    break;

                case 135:
                    if (isFirstMax(magArraylocal[i * NXPROB + j], magArraylocal[(i + 1) * NXPROB + j - 1],
                                   magArraylocal[(i - 1) * NXPROB + j + 1])) {
                        xlocal[0][i][j] = magArraylocal[i * NXPROB + j]; // black
                    }else{
                        xlocal[0][i][j]=0;
                    }
                    break;

                default:
                    //	cout << "error in nomax()"<< endl;
                    break;
            }
        }
    }
    ///---------------------------------value exchange-----------------------------------
    /* Send up unless I'm at the top, then receive from below */
    if (rank < size - 1)
        MPI_Send(xlocal[0][NXPROB / size], NXPROB, MPI_INT, rank + 1, 0,
                 MPI_COMM_WORLD);//012 send to 123
    if (rank > 0)
        MPI_Recv(xlocal[0][0], NXPROB, MPI_INT, rank - 1, 0,
                 MPI_COMM_WORLD, &status);//recv from 012
    /* Send down unless I'm at the bottom */
    if (rank > 0)
        MPI_Send(xlocal[0][1], NXPROB, MPI_INT, rank - 1, 1,
                 MPI_COMM_WORLD);//123 send to 012
    if (rank < size - 1)
        MPI_Recv(xlocal[0][NXPROB / size + 1], NXPROB, MPI_INT, rank + 1, 1,
                 MPI_COMM_WORLD, &status);//recv from 123
    ///-------------------------------Hysteresis-----------------------------------------
    bool greaterFound;
    bool betweenFound;

        for (int i = i_first; i <= i_last; i++) {
            for (int j = 1; j <= NXPROB - 1; j++) {
                if (xlocal[0][i][j] < T_LOW) {
                    xlocal[1][i][j] = 0; // white
                }

                if (xlocal[0][i][j] > T_HIGH) {
                    xlocal[1][i][j] = 255; // black
                }

                /*If pixel (x, y) has gradient magnitude between tlow and thigh and
              any of its neighbors in a 3 * 3 region around
              it have gradient magnitudes greater than thigh, keep the edge*/

                if (xlocal[0][i][j] >= T_LOW && xlocal[0][i][j] <= T_HIGH) {
                    greaterFound = false;
                    betweenFound = false;
                    for (int m = -1; m < 2; m++) {
                        for (int n = -1; n < 2; n++) {
                            if (xlocal[0][i + m][j + n] > T_HIGH) {
                                xlocal[1][i][j] = 0;
                                greaterFound = true;
                            }
                            if (xlocal[0][i][j] > T_LOW && xlocal[0][i][j] < T_HIGH) betweenFound = true;
                        }
                    }

                    if (!greaterFound && betweenFound) {
                        for (int m = -2; m < 3; m++) {
                            for (int n = -2; n < 3; n++) {
                                if (magArraylocal[(i + m) * NXPROB + j + n] > T_HIGH) greaterFound = true;
                            }
                        }
                    }

                    if (greaterFound) xlocal[1][i][j] = 255;
                    else xlocal[1][i][j] = 0;

                }

            }
        }

    ///----------------------------------data collect------------------------------------
    MPI_Gather(&xlocal[1][0][NXPROB], NXPROB * (NXPROB / size), MPI_INT,
               &imageArray[0], NXPROB * (NXPROB / size), MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        T2 = MPI_Wtime(); /* parallel end time */
        int byte;
        BMP OutputIMG;
        OutputIMG.SetBitDepth(DEPTH);
        OutputIMG.SetSize(COLUMNS, ROWS);

        for (int j = 0; j < ROWS; j++) {
            for (int i = 0; i < COLUMNS; i++) {
                byte = imageArray[i + j * COLUMNS];
                OutputIMG(i, j)->Red = byte;
                OutputIMG(i, j)->Green = byte;
                OutputIMG(i, j)->Blue = byte;
            }
        }

        OutputIMG.WriteToFile("Output.bmp");
        cout << T2 - T1 << endl;
        cout << "\n**** NOW GO OPEN Output.BMP ******" << endl;
        delete[] imageArray;
        delete[] thetaslocal;
        delete[] magArraylocal;
    }

    MPI_Finalize();

    return 0;
}