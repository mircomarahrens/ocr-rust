use axum::{
    routing::{get, post},
    Router, Json, Extension,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use base64::prelude::*;
use rocr::nets::lenet::{LeNet, MNIST_MEAN, MNIST_STD};
use rocr::math::tensor::Tensor;
use rocr::math::loss::Softmax;

#[derive(Parser)]
struct ServerArgs {
    #[arg(long)]
    weights_dir: PathBuf,
    #[arg(long, default_value = "3000")]
    port: u16,
}

#[derive(Deserialize)]
struct PredictRequest {
    image: String,
}

#[derive(Serialize)]
struct PredictResponse {
    prediction: usize,
    confidence: f32,
    probabilities: Vec<f32>,
}

fn argmax_1d(tensor: &Tensor<f32>) -> usize {
    let mut max_idx = 0usize;
    let mut max_val = tensor[&[0]];
    for i in 1..10 {
        if tensor[&[i]] > max_val {
            max_val = tensor[&[i]];
            max_idx = i;
        }
    }
    max_idx
}

fn preprocess_base64_image(base64_str: &str) -> Result<Tensor<f32>, String> {
    // Strip base64 header like "data:image/png;base64," if present
    let comma_idx = base64_str.find(',').unwrap_or(0);
    let raw_base64 = if comma_idx > 0 {
        &base64_str[comma_idx + 1..]
    } else {
        base64_str
    };

    // Decode base64 string
    let image_bytes = BASE64_STANDARD.decode(raw_base64.trim())
        .map_err(|e| format!("Base64 decode error: {}", e))?;

    // Load image from decoded bytes
    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| format!("Image load error: {}", e))?;

    // Convert to grayscale (Luma8)
    let grayscale = img.to_luma8();

    // Resize image to 28x28 using smooth bilinear filtering
    let resized = image::imageops::resize(
        &grayscale,
        28,
        28,
        image::imageops::FilterType::Triangle,
    );

    // Normalize image using MNIST mean and standard deviation
    let mut normalized_data = Vec::with_capacity(28 * 28);
    for &pixel in resized.as_raw() {
        let val_01 = (pixel as f32) / 255.0;
        let norm_val = (val_01 - MNIST_MEAN) / MNIST_STD;
        normalized_data.push(norm_val);
    }

    Ok(Tensor::new(normalized_data, vec![28, 28, 1]))
}

async fn handle_index() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("index.html"))
}

async fn handle_predict(
    Extension(model): Extension<Arc<Mutex<LeNet>>>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (axum::http::StatusCode, String)> {
    let mut input_tensor = preprocess_base64_image(&payload.image)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;

    let mut model_guard = model.lock().await;
    // Set model to evaluation mode (disable dropout)
    model_guard.set_training(false);
    let logits = model_guard.forward(&mut input_tensor);
    let predictions = Softmax::forward(&logits);

    let max_idx = argmax_1d(&predictions);
    let confidence = predictions.get_data()[max_idx];

    Ok(Json(PredictResponse {
        prediction: max_idx,
        confidence,
        probabilities: predictions.get_data().clone(),
    }))
}

#[tokio::main]
async fn main() {
    let args = ServerArgs::parse();

    if !args.weights_dir.exists() {
        eprintln!("Error: Weights directory {:?} does not exist.", args.weights_dir);
        std::process::exit(1);
    }

    println!("Loading LeNet model weights from {:?}", args.weights_dir);
    let mut model = LeNet::new();
    if let Err(e) = model.load_weights(&args.weights_dir) {
        eprintln!("Error loading model weights: {}", e);
        std::process::exit(1);
    }
    println!("Model weights loaded successfully!");

    let shared_model = Arc::new(Mutex::new(model));

    let app = Router::new()
        .route("/", get(handle_index))
        .route("/predict", post(handle_predict))
        .layer(Extension(shared_model));

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    println!("OCR Inference Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
