use anyhow::{Error, Result};
use aws_sdk_s3::Client;
use std::env;
use candle_core::Tensor;

/// Attempt to save image to S3 storage, but don't fail the operation if S3 is unavailable
pub async fn save_to_s3_safe(image: &Tensor, uuid: &str) -> Result<(), String> {
    match S3Storage::from_env().await {
        Ok(s3_client) => {
            match s3_client.save_image(image, &format!("{}.jpg", uuid)).await {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("S3 save failed: {}", e)),
            }
        }
        Err(e) => Err(format!("S3 client creation failed: {}", e)),
    }
}

/// S3 storage configuration and client
pub struct S3Storage {
    client: Client,
    bucket_name: String,
    endpoint_url: Option<String>,
}

impl S3Storage {
    /// Create a new S3Storage instance from environment variables
    pub async fn from_env() -> Result<Self> {
        let bucket_name = env::var("S3_BUCKET_NAME")
            .map_err(|_| anyhow::Error::msg("S3_BUCKET_NAME environment variable not set"))?;

        let endpoint_url = env::var("S3_ENDPOINT_URL").ok();
        let access_key = env::var("S3_ACCESS_KEY").ok();
        let secret_key = env::var("S3_SECRET_KEY").ok();

        // Configure AWS SDK
        let mut config_builder = aws_config::from_env();

        if let Some(url) = &endpoint_url {
            config_builder = config_builder.endpoint_url(url);
        }

        // Add credentials if provided
        if let (Some(access_key), Some(secret_key)) = (&access_key, &secret_key) {
            use aws_sdk_s3::config::Credentials;
            let credentials = Credentials::new(access_key, secret_key, None, None, "env");
            config_builder = config_builder.credentials_provider(credentials);
        }

        let config = config_builder.load().await;
        let client = Client::new(&config);

        Ok(Self {
            client,
            bucket_name,
            endpoint_url,
        })
    }

    /// Save an image tensor to S3 storage
    pub async fn save_image(&self, image: &Tensor, object_key: &str) -> Result<()> {
        use std::io::Cursor;

        // Following the same approach as candle_examples::save_image
        let (channel, height, width) = image.dims3()?;
        if channel != 3 {
            anyhow::bail!("save_image expects an input of shape (3, height, width)")
        }

        // Permute from (C, H, W) to (H, W, C) and get pixels
        let img = image.permute((1, 2, 0))?.flatten_all()?;
        let pixels = img.to_vec1::<u8>()?;

        // Create image from raw pixels (same as in candle_examples::save_image)
        let image_buffer: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
                Some(image) => image,
                None => anyhow::bail!("error creating image from tensor"),
            };

        // Convert image to JPEG bytes in memory
        let mut buffer: Vec<u8> = Vec::new();
        image_buffer.write_to(&mut Cursor::new(&mut buffer), image::ImageFormat::Jpeg)
            .map_err(|e| anyhow::Error::msg(format!("error encoding image: {}", e)))?;

        // Upload to S3-compatible storage
        self.client
            .put_object()
            .bucket(&self.bucket_name)
            .key(object_key)
            .body(buffer.into())
            .send()
            .await?;

        // Log the URL where the image was saved
        let s3_url = if let Some(url) = &self.endpoint_url {
            format!("{} -> s3://{}/{}", url, self.bucket_name, object_key)
        } else {
            format!("s3://{}/{}", self.bucket_name, object_key)
        };

        println!("Image saved to S3-compatible storage: {}", s3_url);
        Ok(())
    }
}
