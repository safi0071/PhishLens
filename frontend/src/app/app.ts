import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

type FeatureMap = Record<string, number | null>;

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  apiUrl = 'http://127.0.0.1:5000';
  jsonInput = '';
  result: any = null;
  error = '';
  busy = false;

  constructor(private http: HttpClient) {}

  loadExample() {
    this.error = '';
    this.result = null;
    this.busy = true;
    this.http.get<any>(`${this.apiUrl}/sample`).subscribe({
      next: (res) => {
        this.jsonInput = JSON.stringify(res.features, null, 2);
        this.busy = false;
      },
      error: (err) => {
        this.error = err.error?.error || 'Could not connect to Flask. Start the backend and train the model first.';
        this.busy = false;
      }
    });
  }

  predict() {
    this.error = '';
    this.result = null;
    let features: FeatureMap;
    try {
      features = JSON.parse(this.jsonInput);
      if (!features || Array.isArray(features) || typeof features !== 'object') {
        throw new Error('Input must be a JSON object of feature names and numeric values.');
      }
    } catch (e: any) {
      this.error = e.message || 'Invalid JSON.';
      return;
    }

    this.busy = true;
    this.http.post<any>(`${this.apiUrl}/predict`, { features }).subscribe({
      next: (res) => {
        this.result = res;
        this.busy = false;
      },
      error: (err) => {
        this.error = err.error?.error || 'Prediction request failed. Check the Flask API URL and server.';
        this.busy = false;
      }
    });
  }
}
