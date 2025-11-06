// Simple token bucket rate limiter in Node.js
class TokenBucket {
  constructor(capacity, refillRatePerSec) {
    this.capacity = capacity;
    this.refillRatePerSec = refillRatePerSec;
    this.available = capacity;
    this.lastRefill = Date.now();
  }
  allow(n = 1) {
    const now = Date.now();
    const elapsedSec = (now - this.lastRefill) / 1000;
    if (elapsedSec >= 1) {
      const refillTokens = Math.floor(elapsedSec) * this.refillRatePerSec;
      if (refillTokens > 0) {
        this.available = Math.min(this.capacity, this.available + refillTokens);
        this.lastRefill = now;
      }
    }
    if (n <= this.available) {
      this.available -= n;
      return true;
    }
    return false;
  }
}

const bucket = new TokenBucket(10, 5);
let i = 0;
const interval = setInterval(() => {
  const allowed = bucket.allow();
  console.log(`Request ${i} allowed=${allowed} remaining=${bucket.available}`);
  i++;
  if (i >= 25) {
    clearInterval(interval);
    console.log('Done');
  }
}, 200);
