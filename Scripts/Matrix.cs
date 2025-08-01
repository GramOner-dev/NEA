using System;

public class Matrix
{
    private float[,] matrix;
    private int rowNum, colNum;

    #region ConstructorOverrides
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

    public Matrix(float[,] matrix)
    {
        this.matrix = matrix;
    }

    public Matrix()
    {
        this.matrix = new float[1, 1];
    }
    #endregion

    #region arrayAccessors
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
    #endregion

    #region operatorOverrides
    public static Matrix operator *(Matrix a, Matrix b)
    {

        int aRowNum = a.GetLength(0);
        int bRowNum = b.GetLength(0);
        int aColNum = a.GetLength(1);
        int bColNum = b.GetLength(1);

        if (aColNum != bRowNum)
            throw new ArgumentException(
                $"Matrix dimension mismatch: A is {aRowNum}x{aColNum} B is {bRowNum}x{bColNum} whcih is invalid for multiplication");

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
                $"Matrix dimension mismatch: A is {aRowNum}x{aColNum} B is {bRowNum}x{bColNum} which is invalid for addition");

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
    #endregion

    #region matrixFunctions
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

    public Matrix RowWiseSum()
    {
        Matrix result = new Matrix(rowNum, 1);

        for (int i = 0; i < rowNum; i++)
        {
            float sum = 0f;
            for (int j = 0; j < colNum; j++)
                sum += matrix[i, j];
            result[i, 0] = sum;
        }
        return result;
    }

    public void SetMatrix(Matrix source)
    {
        if (GetLength(0) != source.GetLength(0) || GetLength(1) != source.GetLength(1))
            throw new ArgumentException("source and target matrices must have the same dimensions.");

        int rows = GetLength(0);
        int cols = GetLength(1);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                this[i, j] = source[i, j];
    }
    public static Matrix BroadcastColumn(Matrix columnVector, int numCols)
    {
        if (columnVector.GetLength(1) != 1)
            throw new ArgumentException("ionput isnt in shape (n, 1) for broadcasting.");

        int numRows = columnVector.GetLength(0);
        Matrix result = new Matrix(numRows, numCols);

        for (int i = 0; i < numRows; i++)
        {
            float value = columnVector[i, 0];
            for (int j = 0; j < numCols; j++)
            {
                result[i, j] = value;
            }
        }

        return result;
    }
    #endregion

    #region matrixDimFunctions
    public int GetLength(int dimension)
    {
        if (dimension == 0) return rowNum;
        if (dimension == 1) return colNum;
        throw new ArgumentException("invalid dimension, use 0 for rows, 1 for columns");
    }

    public (int Rows, int Columns) Shape() => (rowNum, colNum);

    public void PrintShape()
    {
        Console.WriteLine($"shape - {rowNum}, {colNum}");
    }
    #endregion

}
