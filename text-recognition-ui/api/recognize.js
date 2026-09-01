export const config = {
  runtime: 'edge',
};

export default async function handler(req) {
  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 });
  }

  // HF_API_URL and HF_TOKEN will be configured in Vercel Environment Variables
  const API_URL = process.env.HF_API_URL;
  const HF_TOKEN = process.env.HF_TOKEN;

  if (!API_URL) {
    return new Response(JSON.stringify({ error: 'HF_API_URL environment variable is not configured.' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  // Clone headers from the incoming request (preserves multipart boundaries, etc.)
  const headers = new Headers(req.headers);
  // Remove host so fetch can set the correct one for Hugging Face
  headers.delete('host');
  
  if (HF_TOKEN) {
    headers.set('Authorization', `Bearer ${HF_TOKEN}`);
  }

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: headers,
      body: req.body,
      duplex: 'half' // Required for streaming request bodies in Vercel Edge
    });

    return new Response(response.body, {
      status: response.status,
      headers: response.headers
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: `Proxy Error: ${error.message}` }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}
