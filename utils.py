import math

def calculate_optimal_chunks(width, height, max_dimension=400):
  """
  Calculate optimal number of chunks to split the image into
  Returns number of chunks in width and height
  """
  chunks_w = math.ceil(width / max_dimension)
  chunks_h = math.ceil(height / max_dimension)

  # Calculate actual chunk dimensions
  chunk_width = width // chunks_w
  chunk_height = height // chunks_h

  return chunks_w, chunks_h, chunk_width, chunk_height

def split_image_dynamic(image):
  """Split image into optimal chunks without overlap"""
  width, height = image.size
  chunks_w, chunks_h, chunk_width, chunk_height = calculate_optimal_chunks(width, height)

  chunks = []
  positions = []

  for y in range(chunks_h):
      for x in range(chunks_w):
          # Calculate chunk coordinates
          left = x * chunk_width
          top = y * chunk_height
          right = left + chunk_width if x < chunks_w - 1 else width
          bottom = top + chunk_height if y < chunks_h - 1 else height

          # Extract chunk
          chunk = image.crop((left, top, right, bottom))
          chunks.append(chunk)
          positions.append((left, top, right, bottom))

  return chunks, positions