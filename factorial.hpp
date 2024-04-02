#pragma once

long unsigned factorial(unsigned p) {
  if (p == 0) {
    return 1;
  } else {
    return p * factorial(p - 1);
  }
}
