#include <iostream>
#include <stdexcept>
#include <format>
#include <type_traits>


namespace lab3
{
using std::cout;
using std::ostream;
using std::common_type_t;
using std::invalid_argument;
using std::out_of_range;

template <typename T>
class Matrix
{
public:
    // Initialization
    Matrix(size_t rows = 0, size_t cols = 0) 
        : rows(rows), cols(cols), data(nullptr) 
    {
        if (rows > 0 && cols > 0)
            allocate(rows, cols);
    }   

    // Rule of five
    Matrix(const Matrix<T> &other) 
        : rows(other.rows), cols(other.cols) 
    {
        if (other.rows > 0 && other.cols > 0) {
            allocate(other.rows, other.cols);
            for (size_t row = 0; row < this->rows; ++row) 
            {
                for (size_t col = 0; col < this->cols; ++col) 
                {
                    this->data[row][col] = other.data[row][col];
                }
            }
        }
    }

    Matrix(Matrix<T> &&other) 
        : rows(other.rows), cols(other.cols), data(other.data) 
    { 
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    Matrix<T>& operator=(const Matrix<T>& other)
    {
        if (this == &other)
            return *this;
        
        if (!is_rows_cols_equal(other)) 
        {
            free_memory();
            allocate(other.rows, other.cols);
        }

        for (size_t row = 0; row < this->rows; ++row) 
        {
            for (size_t col = 0; col < this->cols; ++col) 
            {
                this->data[row][col] = other.data[row][col];
            }
        }

        return *this;
    }

    Matrix<T>& operator=(Matrix<T> &&other) 
    {
        if (this == &other)
            return *this;
        
        free_memory();
            
        this->rows = other.rows;
        this->cols = other.cols;
        this->data = other.data;

        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;

        return *this;
    };

    ~Matrix() 
    {
        free_memory();
    }

    // add, sub and mul
    template <typename U>
    friend Matrix<U> operator+(const Matrix<U> &m);
    template <typename U>
    friend Matrix<U> operator-(const Matrix<U> &m);

    template <typename U, typename V>
    friend Matrix<common_type_t<U, V>> operator+(const Matrix<U> &m1, const Matrix<V> &m2);
    template <typename U, typename V>
    friend Matrix<common_type_t<U, V>> operator+(const U &scalar, const Matrix<V> &m);
    template <typename U, typename V>
    friend Matrix<common_type_t<U, V>> operator+(const Matrix<U> &m, const V &scalar);

    template <typename U, typename V>
    friend Matrix<common_type_t<U, V>> operator-(const Matrix<U> &m1, const Matrix<V> &m2);
    template <typename U, typename V>
    friend Matrix<common_type_t<U, V>> operator-(const U &scalar, const Matrix<V> &m);
    template <typename U, typename V>
    friend Matrix<common_type_t<U, V>> operator-(const Matrix<U> &m, const V &scalar);

    template <typename U, typename V>
    friend Matrix<common_type_t<U, V>> operator*(const Matrix<U> &m1, const Matrix<V> &m2);
    template <typename U, typename V>
    friend Matrix<common_type_t<U, V>> operator*(const U &scalar, const Matrix<V> &m);
    template <typename U, typename V>
    friend Matrix<common_type_t<U, V>> operator*(const Matrix<U> &m, const V &scalar);

    // Logistic
    template <typename U>
    bool operator==(const Matrix<U> &other) const 
    {
        if (!is_rows_cols_equal(other))
            return false;

        for (size_t row = 0; row < this->rows; ++row) 
        {
            for (size_t col = 0; col < this->cols; ++col) 
            {
                if (this->data[row][col] != other.data[row][col]) return false;
            }
        }
        return true;
    }

    template <typename U>
    bool operator!=(const Matrix<U> &other) const 
    {
        return !(*this == other);
    }

    // Resizing
    void operator++() 
    {
        Matrix<T> new_matrix = Matrix<T>(this->rows + 1, this->cols + 1);
        if (this->data) {
            for (size_t row = 0; row < this->rows; ++row) 
            {
                for (size_t col = 0; col < this->cols; ++col) 
                {
                    new_matrix.data[row + 1][col + 1] = this->data[row][col];
                }
            }
        }
        *this = new_matrix;
    }

    void operator++(int) 
    {
        Matrix<T> new_matrix = Matrix<T>(this->rows + 1, this->cols + 1);
        for (size_t row = 0; row < this->rows; ++row) 
        {
            for (size_t col = 0; col < this->cols; ++col) 
            {
                new_matrix.data[row][col] = this->data[row][col];
            }
        }
        *this = new_matrix;
    }

    void operator--() 
    {
        if (this->rows == 0 || this->cols == 0)
            throw invalid_argument("Matrix dimensions is too low");
        
        Matrix<T> new_matrix = Matrix<T>(this->rows - 1, this->cols - 1);
        for (size_t row = 0; row < this->rows - 1; ++row) 
        {
            for (size_t col = 0; col < this->cols - 1; ++col) 
            {
                new_matrix.data[row][col] = this->data[row + 1][col + 1];
            }
        }
        *this = new_matrix;
    }

    void operator--(int) {
        if (this->rows == 0 || this->cols == 0)
            throw invalid_argument("Matrix dimensions is too low");

        Matrix<T> new_matrix = Matrix<T>(this->rows - 1, this->cols - 1);
        for (size_t row = 0; row < this->rows - 1; ++row) 
        {
            for (size_t col = 0; col < this->cols - 1; ++col) 
            {
                new_matrix.data[row][col] = this->data[row][col];
            }
        }
        *this = new_matrix;
    }

    // Printing
    template <typename U>
    friend ostream& operator<<(ostream& os, const Matrix<U>& matrix);
    template <typename U>
    friend ostream& operator>>(ostream& os, Matrix<U>& matrix);

    // Indexing
    T* operator[](const int row) const 
    {
        if (row < 0 || row >= this->rows)
            throw out_of_range("Index out of range");

        return this->data[row];
    }

    T& operator()(const int row, const int col) {
        if (row < 0 || row >= this->rows || col < 0 || col >= this->cols)
            throw out_of_range("Index out of range");

        return this->data[row][col];
    }

    const T& operator()(const int row, const int col) const {
        if (row < 0 || row >= this->rows || col < 0 || col >= this->cols)
            throw out_of_range("Index out of range");

        return this->data[row][col];
    }

    Matrix operator^(const int n) const {
        if (n < 0)
            throw invalid_argument("Power must be non-negative");
        if (this->rows != this->cols)
            throw invalid_argument("Matrix must be square");
        
        if (n == 0) {
            Matrix<T> identity(this->rows, this->cols);
            for (int i = 0; i < this->rows; ++i) {
                identity.data[i][i] = T{1};
            }
            return identity;
        }
        if (n == 1) {
            return *this;
        }
        
        Matrix<T> result = *this;
        for (int i = 1; i < n; ++i) 
        {
            result = result * (*this);
        }
        return result;
    }

    // add, sub and mul in a brand-new way
    template <typename U>
    void operator+=(const Matrix<U> &other);
    template <typename U>
    void operator+=(const U &value);
    template <typename U>
    void operator-=(const Matrix<U> &other);
    template <typename U>
    void operator-=(const U &value);
    template <typename U>
    void operator*=(const Matrix<U> &other);
    template <typename U>
    void operator*=(const U &value);

    //Sume other stuff
    size_t Rows() const {
        return this->rows;
    }
    size_t Cols() const {
        return this->cols;
    }
    Matrix<T> Transpose() const 
    {
    Matrix<T> transposed = Matrix<T>(this->cols, this->rows);
        for (size_t row = 0; row < this->rows; ++row) 
        {
            for (size_t col = 0; col < this->cols; ++col) 
            {
                transposed.data[col][row] = this->data[row][col];
            }
        }
        return transposed;
    }

    template <typename F>
    void apply(F f) {
        for (size_t row = 0; row < this->rows; ++row) 
            for (size_t col = 0; col < this->cols; ++col) 
                this->data[row][col] = f(this->data[row][col]);
    }

    template <typename F, typename U>
    void apply(F f, U value) {
        for (size_t row = 0; row < this->rows; ++row) 
            for (size_t col = 0; col < this->cols; ++col) 
                this->data[row][col] = f(this->data[row][col], value);
    }

    template <typename F, typename U>
    void apply(F f, Matrix<U> m) {
        for (size_t row = 0; row < this->rows; ++row) 
            for (size_t col = 0; col < this->cols; ++col) 
                this->data[row][col] = f(this->data[row][col], m[row][col]);
    }

private:
    T **data;
    size_t rows, cols;

    template <typename U>
    bool is_rows_cols_equal(const Matrix<U> &other) const 
    {
        return this->rows == other.rows && this->cols == other.cols;
    }

    void free_memory() {
        if (!this->data) return;
        for (size_t row = 0; row < this->rows; ++row) 
        {
            delete[] this->data[row];
        }
        delete[] this->data;
        this->data = nullptr;
        this->rows = 0;
        this->cols = 0;
    }

    void allocate(size_t rows, size_t cols) {
        this->rows = rows;
        this->cols = cols;
        this->data = new T*[rows];
        for (size_t row = 0; row < rows; ++row)
            this->data[row] = new T[cols]{};
    }
};


template <typename U>
Matrix<U> operator+(const Matrix<U> &m) {
    return m;
}

template <typename U>
Matrix<U> operator-(const Matrix<U> &m) {
    Matrix<U> result = m;
    result.apply([](auto a) { return -a; });
    return result;
}



template <typename T, typename U>
Matrix<common_type_t<T, U>> operator+(const Matrix<T> &m1, const Matrix<U> &m2) {
    if (m1.Rows() != m2.Rows() || m1.Cols() != m2.Cols())
        throw std::invalid_argument("Matrix dimensions must match");
    Matrix<common_type_t<T, U>> result = m1;
    result.apply([](auto a, auto b) { return a + b; }, m2);
    return result;
}

template <typename T, typename U>
Matrix<common_type_t<T, U>> operator+(const T &scalar, const Matrix<U> &m) {
    Matrix<common_type_t<T, U>> result = m;
    result.apply([scalar](auto a) { return scalar + a; });
    return result;
}

template <typename T, typename U>
Matrix<common_type_t<T, U>> operator+(const Matrix<T> &m, const U &scalar) {
    return scalar + m;
}



template <typename T, typename U>
Matrix<common_type_t<T, U>> operator-(const Matrix<T> &m1, const Matrix<U> &m2) {
    if (m1.Rows() != m2.Rows() || m1.Cols() != m2.Cols())
        throw std::invalid_argument("Matrix dimensions must match");
    Matrix<common_type_t<T, U>> result = m1;
    result.apply([](auto a, auto b) { return a - b; }, m2);
    return result;
}

template <typename T, typename U>
Matrix<common_type_t<T, U>> operator-(const T &scalar, const Matrix<U> &m) {
    Matrix<common_type_t<T, U>> result(m.Rows(), m.Cols());
    result.apply([scalar, &m](auto) { return scalar; });
    result.apply([](auto a, auto b) { return a - b; }, m);
    return result;
}

template <typename T, typename U>
Matrix<common_type_t<T, U>> operator-(const Matrix<T> &m, const U &scalar) {
    Matrix<common_type_t<T, U>> result = m;
    result.apply([scalar](auto a) { return a - scalar; });
    return result;
}



template <typename T, typename U>
Matrix<common_type_t<T, U>> operator*(const Matrix<T> &m1, const Matrix<U> &m2) {
    if (m1.Cols() != m2.Rows())
        throw std::invalid_argument("Matrix dimensions must match for multiplication");
    
    using ResultType = common_type_t<T, U>;
    Matrix<ResultType> result(m1.Rows(), m2.Cols());
    for (size_t row = 0; row < m1.Rows(); ++row) {
        for (size_t col = 0; col < m2.Cols(); ++col) {
            for (size_t k = 0; k < m1.Cols(); ++k) {
                result[row][col] += m1[row][k] * m2[k][col];
            }
        }
    }
    return result;
}

template <typename T, typename U>
Matrix<common_type_t<T, U>> operator*(const T &scalar, const Matrix<U> &m) {
    Matrix<common_type_t<T, U>> result = m;
    result.apply([scalar](auto a) { return scalar * a; });
    return result;
}

template <typename T, typename U>
Matrix<common_type_t<T, U>> operator*(const Matrix<T> &m, const U &scalar) {
    return scalar * m;
}



template <typename U>
ostream& operator<<(ostream& os, const Matrix<U>& matrix) 
{
    for (int row = 0; row < matrix.rows; ++row) 
    {
        for (int col = 0; col < matrix.cols; ++col) 
        {
            os << matrix.data[row][col] << " ";
        }
        os << "\n";
    }
    return os;
}

template <typename U>
ostream& operator>>(ostream& os, Matrix<U>& matrix) 
{
    os << matrix.rows << " " << matrix.cols << "\n";
    for (size_t row = 0; row < matrix.rows; ++row) 
    {
        for (size_t col = 0; col < matrix.cols; ++col) 
        {
            os << matrix.data[row][col] << " ";
        }
    }
    return os;
}



template <typename T>
template <typename U>
void Matrix<T>::operator+=(const Matrix<U> &other) {
    if (!is_rows_cols_equal(other))
        throw std::invalid_argument("Matrix dimensions must match");
    apply([](T a, U b) { return a + b; }, other);
}

template <typename T>
template <typename U>
void Matrix<T>::operator+=(const U &value) {
    apply([](T a, U b) { return a + b; }, value);
}

template <typename T>
template <typename U>
void Matrix<T>::operator-=(const Matrix<U> &other) {
    if (!is_rows_cols_equal(other))
        throw std::invalid_argument("Matrix dimensions must match");
    apply([](T a, U b) { return a - b; }, other);
}

template <typename T>
template <typename U>
void Matrix<T>::operator-=(const U &value) {
    apply([](T a, U b) { return a - b; }, value);
}

template <typename T>
template <typename U>
void Matrix<T>::operator*=(const Matrix<U> &other) {
    *this = *this * other;
}

template <typename T>
template <typename U>
void Matrix<T>::operator*=(const U &value) {
    apply([](T a, U b) { return a * b; }, value);
}



int main();
}

int lab3::main() {

    using std::cout;
    using std::endl;

    cout << "=== Matrix Class Testing Program ===" << endl << endl;
    
    try {
        // Test 1: Default Constructor
        cout << "Test 1: Default Constructor" << endl;
        Matrix<int> m1;
        cout << "Default matrix dimensions: " << m1.Rows() << "x" << m1.Cols() << endl << endl;
        
        // Test 2: Parameterized Constructor
        cout << "Test 2: Parameterized Constructor" << endl;
        Matrix<int> m2(3, 4);
        cout << "Matrix m2(3,4):" << endl << m2 << endl;
        
        // Test 3: Copy Constructor
        cout << "Test 3: Copy Constructor" << endl;
        Matrix<int> m3 = m2;
        cout << "Matrix m3 (copy of m2):" << endl << m3 << endl;
        
        // Test 4: Fill matrix with values
        cout << "Test 4: Fill matrix with values" << endl;
        for (int i = 0; i < m2.Rows(); ++i) {
            for (int j = 0; j < m2.Cols(); ++j) {
                m2(i, j) = i * m2.Cols() + j + 1;
            }
        }
        cout << "Matrix m2 filled with values:" << endl << m2 << endl;
        
        // Test 5: Assignment Operator
        cout << "Test 5: Assignment Operator" << endl;
        Matrix<int> m4;
        m4 = m2;
        cout << "Matrix m4 (assigned from m2):" << endl << m4 << endl;
        
        // Test 6: Move Constructor
        cout << "Test 6: Move Constructor" << endl;
        Matrix<int> m5 = Matrix<int>(2, 3);
        for (int i = 0; i < m5.Rows(); ++i) {
            for (int j = 0; j < m5.Cols(); ++j) {
                m5(i, j) = (i + 1) * 10 + (j + 1);
            }
        }
        cout << "Matrix m5 (moved):" << endl << m5 << endl;
        
        // Test 7: Arithmetic Operations
        cout << "Test 7: Arithmetic Operations" << endl;
        Matrix<int> m6(2, 2);
        Matrix<int> m7(2, 2);
        
        // Fill matrices
        m6(0, 0) = 1; m6(0, 1) = 2;
        m6(1, 0) = 3; m6(1, 1) = 4;
        
        m7(0, 0) = 5; m7(0, 1) = 6;
        m7(1, 0) = 7; m7(1, 1) = 8;
        
        cout << "Matrix m6:" << endl << m6 << endl;
        cout << "Matrix m7:" << endl << m7 << endl;
        
        // Addition
        Matrix<int> m8 = m6 + m7;
        cout << "m6 + m7:" << endl << m8 << endl;
        
        // Subtraction
        Matrix<int> m9 = m7 - m6;
        cout << "m7 - m6:" << endl << m9 << endl;
        
        // Scalar operations
        Matrix<int> m10 = m6 + 10;
        cout << "m6 + 10:" << endl << m10 << endl;
        
        Matrix<int> m11 = m6 * 2;
        cout << "m6 * 2:" << endl << m11 << endl;
        
        // Test 8: Matrix Multiplication
        cout << "Test 8: Matrix Multiplication" << endl;
        Matrix<int> m12(2, 3);
        Matrix<int> m13(3, 2);
        
        // Fill m12
        for (int i = 0; i < m12.Rows(); ++i) {
            for (int j = 0; j < m12.Cols(); ++j) {
                m12(i, j) = i * m12.Cols() + j + 1;
            }
        }
        
        // Fill m13
        for (int i = 0; i < m13.Rows(); ++i) {
            for (int j = 0; j < m13.Cols(); ++j) {
                m13(i, j) = (i + 1) * 2 + j;
            }
        }
        
        cout << "Matrix m12 (2x3):" << endl << m12 << endl;
        cout << "Matrix m13 (3x2):" << endl << m13 << endl;
        
        Matrix<int> m14 = m12 * m13;
        cout << "m12 * m13:" << endl << m14 << endl;
        
        // Test 9: Comparison Operators
        cout << "Test 9: Comparison Operators" << endl;
        Matrix<int> m15 = m6;
        cout << "m6 == m15: " << (m6 == m15 ? "true" : "false") << endl;
        cout << "m6 == m7: " << (m6 == m7 ? "true" : "false") << endl;
        cout << "m6 != m7: " << (m6 != m7 ? "true" : "false") << endl << endl;
        
        // Test 10: Increment/Decrement Operators
        cout << "Test 10: Increment/Decrement Operators" << endl;
        Matrix<int> m16(2, 2);
        m16(0, 0) = 1; m16(0, 1) = 2;
        m16(1, 0) = 3; m16(1, 1) = 4;
        
        cout << "Original m16:" << endl << m16 << endl;
        cout << "Dimensions: " << m16.Rows() << "x" << m16.Cols() << endl;
        
        ++m16;
        cout << "After ++m16:" << endl << m16 << endl;
        cout << "Dimensions: " << m16.Rows() << "x" << m16.Cols() << endl;
        
        --m16;
        cout << "After --m16:" << endl << m16 << endl;
        cout << "Dimensions: " << m16.Rows() << "x" << m16.Cols() << endl << endl;
        
        // Test 11: Transpose
        cout << "Test 11: Transpose" << endl;
        Matrix<int> m17(2, 3);
        for (int i = 0; i < m17.Rows(); ++i) {
            for (int j = 0; j < m17.Cols(); ++j) {
                m17(i, j) = i * m17.Cols() + j + 1;
            }
        }
        cout << "Original m17:" << endl << m17 << endl;
        cout << "Transposed m17:" << endl << m17.Transpose() << endl << endl;
        
        // Test 12: Power Operator
        cout << "Test 12: Power Operator" << endl;
        Matrix<int> m18(2, 2);
        m18(0, 0) = 1; m18(0, 1) = 1;
        m18(1, 0) = 1; m18(1, 1) = 0;
        
        cout << "Matrix m18:" << endl << m18 << endl;
        Matrix<int> m19 = m18^2;
        cout << "m18^2:" << endl << m19 << endl;
        Matrix<int> m20 = m18^3;
        cout << "m18^3:" << endl << m20 << endl << endl;
        
        // Test 13: Apply Function
        cout << "Test 13: Apply Function" << endl;
        Matrix<int> m21(2, 2);
        m21(0, 0) = 1; m21(0, 1) = 2;
        m21(1, 0) = 3; m21(1, 1) = 4;
        
        cout << "Original m21:" << endl << m21 << endl;
        
        // Test 14: Compound Assignment Operators
        cout << "Test 14: Compound Assignment Operators" << endl;
        Matrix<int> m22(2, 2);
        Matrix<int> m23(2, 2);
        
        m22(0, 0) = 1; m22(0, 1) = 2;
        m22(1, 0) = 3; m22(1, 1) = 4;
        
        m23(0, 0) = 1; m23(0, 1) = 1;
        m23(1, 0) = 1; m23(1, 1) = 1;
        
        cout << "m22:" << endl << m22 << endl;
        cout << "m23:" << endl << m23 << endl;
        
        m22 += m23;
        cout << "After m22 += m23:" << endl << m22 << endl;
        
        m22 -= m23;
        cout << "After m22 -= m23:" << endl << m22 << endl;
        
        m22 *= 2;
        cout << "After m22 *= 2:" << endl << m22 << endl << endl;
        
        // Test 15: Access Operators
        cout << "Test 15: Access Operators" << endl;
        Matrix<int> m24(2, 2);
        m24(0, 0) = 10; m24(0, 1) = 20;
        m24(1, 0) = 30; m24(1, 1) = 40;
        
        cout << "Matrix m24:" << endl << m24 << endl;
        cout << "m24[0][1] = " << m24[0][1] << endl;
        cout << "m24(1, 0) = " << m24(1, 0) << endl << endl;
        
        // Test 16: Error Handling
        cout << "Test 16: Error Handling" << endl;
        try {
            Matrix<int> m25(2, 2);
            Matrix<int> m26(3, 3);
            Matrix<int> m27 = m25 + m26; // Should throw exception
        } catch (const invalid_argument& e) {
            cout << "Caught expected exception: " << e.what() << endl;
        }
        
        try {
            Matrix<int> m28(2, 2);
            Matrix<int> m29(3, 2);
            Matrix<int> m30 = m28 * m29; // Should throw exception
        } catch (const invalid_argument& e) {
            cout << "Caught expected exception: " << e.what() << endl;
        }
        
        try {
            Matrix<int> m31(1, 1);
            --m31; // Should throw exception
        } catch (const invalid_argument& e) {
            cout << "Caught expected exception: " << e.what() << endl;
        }
        
        // Test 17: Memory Leak Test
        cout << "Test 17: Memory Leak Test" << endl;
        {
            Matrix<int> temp(1000, 1000);
            for (int i = 0; i < 100; ++i) {
                Matrix<int> copy = temp;
                Matrix<int> moved = std::move(copy);
            }
        }
        cout << "Large matrix operations completed" << endl << endl;
        
        cout << endl << "=== All tests completed successfully! ===" << endl;
        
    } catch (const std::exception& e) {
        cout << "Unexpected error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}

int main() {
    return lab3::main();
}