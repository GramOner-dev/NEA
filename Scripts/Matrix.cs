using System;

public class Matrix
{
    private float[,] matrix;
    private int rowNum, colNum;
    public Matrix(int rowNum, int colNum)
    {
        this.matrix = new float[rowNum, colNum];
        this.rowNum = rowNum;
        this.colNum = colNum;
    }

    public Matrix(int rowNum)
    {
        this.matrix = new float[rowNum, 1];
        this.rowNum = rowNum;
        this.colNum = 1;
    }

    public Matrix(float[] vector)
    {
        this.rowNum = 1;
        this.colNum = vector.Length;
        this.matrix = new float[rowNum, colNum];

        for (int i = 0; i < colNum; i++)
            matrix[0, i] = vector[i];

    }

    public Matrix(float[,] matrix) => this.matrix = matrix;


    public float this[int row, int col]
    {
        get => matrix[row, col];
        set => matrix[row, col] = value;
    }

    public float[] this[int row]
    {
        get
        {
            float[] result = new float[colNum];
            for (int j = 0; j < colNum; j++)
                result[j] = matrix[row, j];
            return result;
        }
        set
        {
            for (int j = 0; j < colNum; j++)
                matrix[row, j] = value[j];
        }
    }

    public static Matrix operator *(Matrix a, Matrix b)
    {

        int aRowNum = a.GetLength(0);
        int bRowNum = b.GetLength(0);
        int aColNum = a.GetLength(1);
        int bColNum = b.GetLength(1);

        if (aColNum != bRowNum)
            throw new ArgumentException("matrix dimensions must match for multiplication");

        Matrix result = new Matrix(aRowNum, bColNum);

        for (int i = 0; i < aRowNum; i++)
        {
            for (int j = 0; j < bColNum; j++)
            {

                float sum = 0f;

                for (int k = 0; k < aColNum; k++)
                    sum += a[i, k] * b[k, j];

                result[i, j] = sum;
            }
        }
        return result;
    }

    public static Matrix operator +(Matrix a, Matrix b)
    {
        int aRowNum = a.GetLength(0);
        int bRowNum = b.GetLength(0);
        int aColNum = a.GetLength(1);
        int bColNum = b.GetLength(1);

        if (aRowNum != bRowNum || aColNum != bColNum)
            throw new ArgumentException(
                $"Matrix dimension mismatch: A is {aRowNum}x{aColNum}, B is {bRowNum}x{bColNum}. Dimensions must match for addition.");

        Matrix result = new Matrix(aRowNum, aColNum);

        for (int i = 0; i < aRowNum; i++)
        {
            for (int j = 0; j < aColNum; j++)
            {
                result[i, j] = a[i, j] + b[i, j];
            }
        }

        return result;
    }

    public Matrix Apply(Func<float, float> func)
    {
        Matrix result = new Matrix(rowNum, colNum);

        for (int i = 0; i < rowNum; i++)
        {
            for (int j = 0; j < colNum; j++)
            {
                result[i, j] = func(matrix[i, j]);
            }
        }
        return result;
    }

    public void Fill(float value)
    {
        for (int i = 0; i < rowNum; i++)
            for (int j = 0; j < colNum; j++)
                matrix[i, j] = value;
    }

    public Matrix HeInit()
    {
        for (int i = 0; i < rowNum; i++)
        {
            float[] row = MathsUtils.HeInit(colNum);
            for (int j = 0; j < colNum; j++)
            {
                matrix[i, j] = row[j];
            }
        }
        return this;
    }

    public Matrix Transpose()
    {
        Matrix result = new Matrix(colNum, rowNum);
        for (int i = 0; i < rowNum; i++)
        {
            for (int j = 0; j < colNum; j++)
            {
                result[j, i] = matrix[i, j];
            }
        }
        return result;
    }

    public Matrix Hadamard(Matrix matrixB)
    {
        int bRowNum = matrixB.GetLength(0);
        int bColNum = matrixB.GetLength(1);

        if (this.rowNum != bRowNum || this.colNum != bColNum)
            throw new ArgumentException("matrix dimensions must match for additio");

        Matrix result = new Matrix(this.rowNum, this.colNum);
        for (int i = 0; i < rowNum; i++)
        {
            for (int j = 0; j < colNum; j++)
            {
                result[i, j] *= matrixB[i, j];
            }
        }
        return result;
    }

    public int GetLength(int dimension)
    {
        if (dimension == 0) return rowNum;
        if (dimension == 1) return colNum;
        throw new ArgumentException("invalid dimension, use 0 for rows, 1 for columns");
    }

    public (int Rows, int Columns) Shape() => (rowNum, colNum);

    public void PrintShape()
    {
        Console.WriteLine($"Shape: ({rowNum}, {colNum})");
    }
}
