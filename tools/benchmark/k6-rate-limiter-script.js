/* k6 script placeholder for rate limiter load test
   Usage (after installing k6):
   k6 run k6-rate-limiter-script.js
*/
import http from 'k6/http';
import { Rate } from 'k6/metrics';

export const errorRate = new Rate('errors');

export const options = {
  vus: 10,
  duration: '10s'
};

export default function () {
  const res = http.get('http://localhost:8080/allow'); // Replace with actual endpoint when service added
  if (res.status !== 200) {
    errorRate.add(1);
  }
}
