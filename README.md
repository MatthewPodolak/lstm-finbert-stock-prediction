<div align="center">
  <p>
  "Predict the stock market with deep learning" as the initial bachelor's thesis idea. But to avoid building just another Kaggle-tutorial rerun, I wanted to rigorously test a hypothesis: does sentiment provide added value, and does an ensemble of concurrent models bring anything to the table? I built two independent LSTM classifiers operating on different intraday timeframes (5m and 15m), fused their probabilistic outputs through eleven decision strategies, and bolted a pretrained FinBERT on top of some of them to test it.
  </p>
</div>

<br><br>

<p align="center">
  <img src="https://fuszerkomat.ovh/tasks/dotnet.png" width="90" />
  <img src="https://fuszerkomat.ovh/tasks/plus.png" width="60" />
  <img src="https://fuszerkomat.ovh/tasks/python.png" width="90" />
</p>

<br><br>

<h2>Key takeaways</h2>
<p>
    <ul>
    <li>
      <strong>LSTMs beat random chance.</strong> Both sequential models exceeded the baseline random classifier. They achieved 50.43% accuracy on the 5-minute interval and 52.78% on the 15-minute interval, confirming that short-term price direction is learnable, albeit marginally. The baseline for random chance was 33.3% (slightly more since labels weren't equal) in the initial 3-class classification setup. Classifying 'short' moves performed the poorest and proved ineffective. Consequently, inference is now focused on a 2-class setup (long or none).    </li>
    <li>
      <strong>Model fusion didn't help (much).</strong> None of the LSTM fusion strategies consistently outperformed the better of the two base models. Naive signal combination only adds value when both base models are comparably strong.
    </li>
    <li>
      <strong>Positive sentiment helps; negative sentiment is asymmetric.</strong> Integrating FinBERT's positive sentiment signal improved classification quality on a subset of instruments. The negative sentiment veto mechanism showed no symmetric predictive value. Likely a property of how markets absorb bad news rather than proof that negative sentiment carries no signal at all.
    </li>
    <li>
      <strong>Optimism and pessimism are not mirror images.</strong> Market reactions to positive and negative information follow different dynamics. Prediction systems should model them as two independent processes with different sensitivities, not as symmetric forces.
    </li>
    <li>
      <strong>The harder the timeframe, the more it matters.</strong> The 15-minute model consistently outperformed the 5-minute one, suggesting that very short horizons are dominated by noise and that slightly longer intraday windows offer a more tractable signal.
    </li>
    <li>
      <strong>Volatility and price-flow features carry the signal. Momentum indicators are noise.</strong>
    </li>
  </ul>
</p>

<br><br>

<h2>Replication</h2>
<p>
<ol>
  <li>
    <strong>Get the data</strong> - Sign up at <a href="https://eodhd.com/">eodhd.com</a> and pull both <em>stock price</em> and <em>news</em> data for your ticker(s) of choice.
  </li>
  <li>
    <strong>Prepare the data</strong> - Run the .NET data-preparation module located in <code>/modules/data-preparation</code>. This handles filling, htf aggregation, feature engineering, label engineering and dataset splits.
  </li>
  <li>
    <strong>Train &amp; run inference</strong> - Navigate to <code>/modules/pipeline</code>, install dependencies (<code>pip install -r req</code>), and execute the training + inference pipeline. LSTM classifiers (5m and 15m timeframes) will train sequentially; fusion strategies and FinBERT sentiment scoring are applied automatically during inference.<br>
    <br>
    Finbert model used in project was prosusAI/finBERT - get yours from a <a href="https://huggingface.co/ProsusAI/finbert">hugging face</a> and throw safetensors into <code>/models/finbert_model</code> inside python pipeline.
  </li>
</ol>
</p>
