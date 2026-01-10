#include <bits/stdc++.h>

using namespace std;

const long long m = 1e6;
long long a[m + 5];
long long ps[m + 5];

int main() {
  long long n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  ps[0] = a[0];
  for (int i = 1; i < n; i++) {
    ps[i] = ps[i - 1] + a[i];
  }
  long long tong = ps[n - 1];
  if (tong % 3 != 0) {
    cout << "-1";
    return 0;
  }

  long long tb = tong / 3;
  long long vt1 = -1, vt2 = -1;
  for (int i = 0; i < n - 1; i++) {
    if (ps[i] == tb && vt1 == -1) vt1 = i;
    else if (ps[i] == 2 * tb && vt1 != -1) {
      vt2 = i;
      break;
    }
  }
  if (vt1 != -1 && vt2 != -1 && vt2 < n - 1) cout << vt1 << " " << vt2;
  else cout << "-1";
  return 0;
}