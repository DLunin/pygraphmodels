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
    for (auto& x: arr) 
        cout << x << ' ';
    cout << endl;
    return 0;
}
