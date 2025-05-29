import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, List, Set, Tuple

import aiohttp
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

# SupaBase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Use service role key to bypass RLS
BUCKET_NAME = "audio-generations"

# API Configuration
MODEL_A_API_URL = os.getenv("MODEL_A_API_URL", "https://stable-audio-tools-synt-2-api-pretrained--e49ffc26d5.jobs.imdc.org.neu.ro")
MODEL_B_API_URL = os.getenv("MODEL_B_API_URL", "https://stable-audio-tools-synt-2-api--e49ffc26d5.jobs.imdc.org.neu.ro")
MAX_CONCURRENT_CALLS = 4  # Total concurrent calls (legacy, kept for compatibility)
MAX_CONCURRENT_CALLS_PER_MODEL = 4  # New setting for per-model concurrency
PROMPTS_FILE = "prompts/test_prompts.txt"
OUTPUT_DIR = "test_audio_outputs"
MAX_RETRIES = 3  # Maximum number of retries for failed generations
RETRY_DELAY = 5  # Seconds to wait between retries

# Model identifiers
MODEL_A_ID = "model-a"
MODEL_B_ID = "model-b"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, MODEL_A_ID), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, MODEL_B_ID), exist_ok=True)

# Initialize Supabase client with service role key
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials. Please check your .env file.")

print(f"Initializing Supabase client with URL: {SUPABASE_URL}")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def generate_audio(
    session: aiohttp.ClientSession, 
    prompt: str, 
    index: int, 
    model_id: str,
    model_api_url: str,
    retry_count: int = 0
) -> Tuple[str, int]:
    """Generate audio using the API and return the file path where it was saved and the seed used"""
    
    print(f"Generating audio for prompt {index} with {model_id}: {prompt[:50]}...{' (Retry #'+str(retry_count)+')' if retry_count > 0 else ''}")
    
    # Prepare request payload
    seed = int(time.time()) % 1000000000  # Generate a somewhat random but reproducible seed
    payload = {
        "prompt": prompt,
        "seconds_total": 10.0,  # Using 10 seconds to make it faster - adjust as needed
        "steps": 100,          # Reduced steps for faster generation - adjust as needed
        "cfg_scale": 6.0,
        "seed": seed           # Use the same seed for both models for fair comparison
    }
    
    try:
        # Make POST request to generate endpoint with timeout
        async with session.post(f"{model_api_url}/generate", json=payload, timeout=60) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"Error generating audio for prompt {index} with {model_id}: {error_text}")
                return None, None
            
            result = await response.json()
            file_url = result["file_url"]
            # Use the seed from the result (might be different from requested if -1 was used)
            actual_seed = result["seed"]
            
            # Make GET request to download the generated audio
            model_output_dir = os.path.join(OUTPUT_DIR, model_id)
            local_file_path = os.path.join(model_output_dir, f"prompt_{index:03d}_seed_{actual_seed}.wav")
            async with session.get(f"{model_api_url}{file_url}", timeout=60) as download_response:
                if download_response.status != 200:
                    print(f"Error downloading audio for prompt {index} with {model_id}")
                    return None, None
                
                # Save the audio file locally
                with open(local_file_path, "wb") as f:
                    audio_data = await download_response.read()
                    f.write(audio_data)
            
            # Generate a unique filename for SupaBase
            filename = f"{model_id}_prompt_{index:03d}_seed_{actual_seed}.wav"
            storage_path = f"test-generations/{model_id}/{filename}"
            
            try:
                # Upload to SupaBase
                with open(local_file_path, "rb") as f:
                    upload_result = supabase.storage.from_(BUCKET_NAME).upload(
                        path=storage_path,
                        file=f,
                        file_options={"content-type": "audio/wav"}
                    )
                    print(f"Upload success for {storage_path}")
                
                # Create record in the database with is_public=true
                generation_data = {
                    "prompt": prompt,
                    "negative_prompt": None,
                    "seed": actual_seed,
                    "cfg_scale": payload["cfg_scale"],
                    "steps": payload["steps"],
                    "seconds_total": payload["seconds_total"],
                    "filename": filename,
                    "storage_path": storage_path,
                    "model_version": model_id,
                    "is_public": True,  # Mark as public test generation
                    "test_index": index  # Add index for easier pairing in the UI
                }
                
                # Insert record into database
                db_result = supabase.table("audio_generations").insert(generation_data).execute()
                
                print(f"✓ Successfully generated, uploaded and recorded audio for prompt {index} with {model_id}")
                return local_file_path, actual_seed
            except Exception as upload_err:
                print(f"Supabase error for prompt {index} with {model_id}: {str(upload_err)}")
                return local_file_path, actual_seed  # Return the local path since we still generated the audio
    
    except asyncio.TimeoutError:
        print(f"Timeout error for prompt {index} with {model_id}. The request took too long to complete.")
        return None, None
    except Exception as e:
        print(f"Exception during audio generation for prompt {index} with {model_id}: {str(e)}")
        return None, None

async def process_prompts(prompts: List[str]) -> Dict[str, Set[Tuple[int, str]]]:
    """Process all prompts with both models concurrently"""
    
    async with aiohttp.ClientSession() as session:
        # Create semaphores to limit concurrent API calls per model
        # Each model gets its own concurrency limit since they're on separate GPUs
        semaphore_a = asyncio.Semaphore(MAX_CONCURRENT_CALLS_PER_MODEL)
        semaphore_b = asyncio.Semaphore(MAX_CONCURRENT_CALLS_PER_MODEL)
        
        # Dictionary to track results and seeds
        prompt_seeds = {}
        
        async def bounded_generate_model_a(prompt: str, index: int):
            async with semaphore_a:
                path, seed = await generate_audio(session, prompt, index, MODEL_A_ID, MODEL_A_API_URL)
                if seed:
                    prompt_seeds[index] = seed
                return (index, prompt, path)
        
        async def bounded_generate_model_b(prompt: str, index: int):
            async with semaphore_b:
                path, seed = await generate_audio(session, prompt, index, MODEL_B_ID, MODEL_B_API_URL)
                return (index, prompt, path)
        
        # Create tasks for both Model A and Model B simultaneously
        tasks_a = [bounded_generate_model_a(prompt, i) for i, prompt in enumerate(prompts)]
        tasks_b = [bounded_generate_model_b(prompt, i) for i, prompt in enumerate(prompts)]
        
        # Run all tasks concurrently - both models will process prompts in parallel
        all_results = await asyncio.gather(
            asyncio.gather(*tasks_a),  # All Model A tasks
            asyncio.gather(*tasks_b)   # All Model B tasks
        )
        
        # Split results by model
        results_a = all_results[0]
        results_b = all_results[1]
        
        # Track results for retry if needed
        failed_a = set()
        failed_b = set()
        
        for idx, prompt, result in results_a:
            if result is None:
                failed_a.add((idx, prompt))
        
        for idx, prompt, result in results_b:
            if result is None:
                failed_b.add((idx, prompt))
        
        print(f"\nFirst pass: Processed {len(prompts)} prompts with two models concurrently.")
        print(f"Model A: Failed to generate {len(failed_a)} audio files.")
        print(f"Model B: Failed to generate {len(failed_b)} audio files.")
        
        return {MODEL_A_ID: failed_a, MODEL_B_ID: failed_b}

async def retry_failed_generations(failed_prompts_by_model: Dict[str, Set[Tuple[int, str]]]) -> Dict[str, Set[Tuple[int, str]]]:
    """Retry failed generations with exponential backoff for each model"""
    
    if not any(failed_prompts_by_model.values()):
        return {MODEL_A_ID: set(), MODEL_B_ID: set()}
    
    remaining_failures = {
        model_id: failed_prompts.copy() 
        for model_id, failed_prompts in failed_prompts_by_model.items()
    }
    
    retry_count = 1
    
    api_urls = {
        MODEL_A_ID: MODEL_A_API_URL,
        MODEL_B_ID: MODEL_B_API_URL
    }
    
    while any(remaining_failures.values()) and retry_count <= MAX_RETRIES:
        print(f"\nRetry attempt #{retry_count} for failed generations...")
        
        # Wait before retrying to allow server to recover
        await asyncio.sleep(RETRY_DELAY * retry_count)
        
        # Process both models concurrently for retries too
        retry_tasks = []
        
        for model_id, failures in remaining_failures.items():
            if not failures:
                continue
                
            print(f"Retrying {len(failures)} failed generations for {model_id}...")
            
            # Create a semaphore with per-model concurrency
            semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_CALLS_PER_MODEL // 2))
            
            async def bounded_retry(item, model_id=model_id, semaphore=semaphore):
                idx, prompt = item
                async with semaphore:
                    path, _ = await generate_audio(
                        session=aiohttp.ClientSession(), 
                        prompt=prompt, 
                        index=idx, 
                        model_id=model_id, 
                        model_api_url=api_urls[model_id], 
                        retry_count=retry_count
                    )
                    return (idx, prompt, path, model_id)
            
            # Create tasks for all failed prompts for this model
            model_tasks = [bounded_retry(item) for item in failures]
            retry_tasks.extend(model_tasks)
        
        if not retry_tasks:
            break
            
        # Wait for all retry tasks to complete
        retry_results = await asyncio.gather(*retry_tasks)
        
        # Process results by model
        for idx, prompt, result, model_id in retry_results:
            if result is not None:
                # Success - remove from failures
                remaining_failures[model_id].discard((idx, prompt))
                print(f"Successfully retried generation for prompt {idx} with {model_id}")
            else:
                # Still failed
                print(f"Still failed to generate for prompt {idx} with {model_id}")
        
        # Print summary for this retry attempt
        for model_id, failures in remaining_failures.items():
            print(f"{model_id}: {len(failures)} failures remaining after retry #{retry_count}")
        
        retry_count += 1
    
    return remaining_failures

def create_test_pairs():
    """Create test pairs in the database for A/B evaluation"""
    
    print("Creating evaluation test pairs in the database...")
    
    # Get all test generations grouped by test_index
    response = supabase.table("audio_generations") \
        .select("id, prompt, storage_path, model_version, test_index") \
        .eq("is_public", True) \
        .execute()
    
    generations = response.data
    
    # Group by test_index
    pairs_by_index = {}
    for gen in generations:
        idx = gen.get("test_index")
        if idx is not None:
            if idx not in pairs_by_index:
                pairs_by_index[idx] = []
            pairs_by_index[idx].append(gen)
    
    # Create pairs for evaluation
    created_pairs = 0
    skipped_pairs = 0
    
    for idx, items in pairs_by_index.items():
        if len(items) < 2:
            print(f"Warning: Only found {len(items)} generation(s) for test index {idx}, skipping")
            skipped_pairs += 1
            continue
            
        # We should have 2 items (model A and model B)
        model_a_gen = next((g for g in items if g["model_version"] == MODEL_A_ID), None)
        model_b_gen = next((g for g in items if g["model_version"] == MODEL_B_ID), None)
        
        if not model_a_gen or not model_b_gen:
            print(f"Warning: Missing model generation for test index {idx}")
            skipped_pairs += 1
            continue
            
        # Create the evaluation pair entry
        test_pair = {
            "prompt": model_a_gen["prompt"],
            "generation_a_id": model_a_gen["id"],
            "generation_b_id": model_b_gen["id"],
            "test_index": idx
        }
        
        try:
            # Check if pair already exists
            existing = supabase.table("evaluation_pairs") \
                .select("id") \
                .eq("test_index", idx) \
                .execute()
                
            if existing.data:
                print(f"Pair for test index {idx} already exists, skipping")
                continue
                
            # Insert new pair
            result = supabase.table("evaluation_pairs").insert(test_pair).execute()
            created_pairs += 1
        except Exception as e:
            print(f"Error creating test pair for index {idx}: {str(e)}")
    
    print(f"Created {created_pairs} evaluation pairs in the database")
    if skipped_pairs > 0:
        print(f"Skipped {skipped_pairs} incomplete pairs")
    
    # Verify created pairs can be accessed properly
    try:
        verify_audio_access()
    except Exception as e:
        print(f"Error verifying audio access: {str(e)}")

def verify_audio_access():
    """Verify that all created pairs have accessible audio files"""
    print("\nVerifying audio file access for created pairs...")
    
    # Get a few pairs to check
    response = supabase.table("evaluation_pairs").select("""
        id, 
        generation_a_id, 
        generation_b_id,
        audio_generations!generation_a_id (storage_path),
        audio_generations!generation_b_id (storage_path)
    """).limit(5).execute()
    
    if not response.data:
        print("No pairs found to verify")
        return
    
    verified_count = 0
    errors_count = 0
    
    for pair in response.data:
        path_a = pair['audio_generations']['storage_path'] if pair.get('audio_generations') else None
        path_b = pair['audio_generations!generation_b_id']['storage_path'] if pair.get('audio_generations!generation_b_id') else None
        
        print(f"Verifying pair {pair['id']}:")
        
        if path_a:
            try:
                # Test if file can be accessed
                url_a = supabase.storage.from_(BUCKET_NAME).create_signed_url(path_a, 60)
                if url_a.data and 'signedUrl' in url_a.data:
                    print(f"✓ Model A file accessible: {path_a}")
                    verified_count += 1
                else:
                    print(f"✗ Model A file not accessible: {path_a}")
                    errors_count += 1
            except Exception as e:
                print(f"✗ Error accessing Model A file: {str(e)}")
                errors_count += 1
        
        if path_b:
            try:
                # Test if file can be accessed
                url_b = supabase.storage.from_(BUCKET_NAME).create_signed_url(path_b, 60)
                if url_b.data and 'signedUrl' in url_b.data:
                    print(f"✓ Model B file accessible: {path_b}")
                    verified_count += 1
                else:
                    print(f"✗ Model B file not accessible: {path_b}")
                    errors_count += 1
            except Exception as e:
                print(f"✗ Error accessing Model B file: {str(e)}")
                errors_count += 1
    
    print(f"Verification complete: {verified_count} files verified, {errors_count} errors")

def read_prompts(file_path: str) -> List[str]:
    """Read prompts from the file, skipping empty lines"""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def save_failed_prompts(
    failed_prompts_by_model: Dict[str, Set[Tuple[int, str]]], 
    output_file: str = "failed_prompts.txt"
):
    """Save failed prompts to a file for manual processing later"""
    with open(output_file, "w") as f:
        for model_id, failed_prompts in failed_prompts_by_model.items():
            if failed_prompts:
                f.write(f"=== Failed for {model_id} ===\n")
                for idx, prompt in sorted(failed_prompts):
                    f.write(f"{idx}: {prompt}\n")
                f.write("\n")
    
    total_failed = sum(len(failed) for failed in failed_prompts_by_model.values())
    print(f"Saved {total_failed} failed prompts to {output_file}")

async def main_async():
    # Read prompts from file
    print(f"Reading prompts from {PROMPTS_FILE}...")
    prompts = read_prompts(PROMPTS_FILE)
    print(f"Found {len(prompts)} prompts.")
    
    # Process prompts
    print(f"Starting audio generation with max {MAX_CONCURRENT_CALLS_PER_MODEL} concurrent API calls per model...")
    start_time = time.time()
    
    # First attempt
    failed_prompts_by_model = await process_prompts(prompts)
    
    # Retry failed generations
    has_failures = any(failed_prompts_by_model.values())
    if has_failures:
        remaining_failures = await retry_failed_generations(failed_prompts_by_model)
        
        if any(remaining_failures.values()):
            print(f"\nAfter {MAX_RETRIES} retry attempts, some generations still failed.")
            save_failed_prompts(remaining_failures)
            
            # Only work with prompts that succeeded for both models
            print("\nIdentifying complete prompt pairs where both models succeeded...")
            model_a_failures = {idx for idx, _ in remaining_failures.get(MODEL_A_ID, set())}
            model_b_failures = {idx for idx, _ in remaining_failures.get(MODEL_B_ID, set())}
            all_failed_indexes = model_a_failures.union(model_b_failures)
            
            if all_failed_indexes:
                print(f"Found {len(all_failed_indexes)} prompts with at least one failed generation")
                print("Removing incomplete generations to ensure only complete pairs are created...")
                
                # Delete generations for prompts that don't have both model A and model B
                for idx in all_failed_indexes:
                    # Find any generations for this prompt index
                    try:
                        response = supabase.table("audio_generations") \
                            .select("id, model_version") \
                            .eq("test_index", idx) \
                            .execute()
                        
                        if response.data:
                            for gen in response.data:
                                print(f"Deleting incomplete generation for prompt {idx}, model {gen['model_version']}")
                                supabase.table("audio_generations").delete().eq("id", gen["id"]).execute()
                    except Exception as e:
                        print(f"Error cleaning up incomplete generation for prompt {idx}: {str(e)}")
        else:
            print("\nAll generations completed successfully after retries!")
    else:
        print("\nAll generations completed successfully on the first attempt!")
    
    # Create test pairs for evaluation
    create_test_pairs()
    
    # Calculate and display elapsed time
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 