#include <iostream>
#include "mdarray.h"

using namespace std;

int main() {
    array_nd<double> arr({2, 5});
    arr[{1, 2}] = 3;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) 
            cout << arr[{i, j}] << ' ';
        cout << endl;
    }
    array_nd<double> arr1({1, 5});
    arr1[{0, 2}] = 5;
    auto res = arr*arr1;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) 
            cout << res[{i, j}] << ' ';
        cout << endl;
    }
    return 0;
}
