using System;
using System.Collections.Generic;

namespace LSTM_DATA_SHAPER.Models
{
    public class TimedStockData
    {
        public string? FolderPath { get; set; }
        public string Ticker { get; set; } = "Unknown.US";
        public List<Candle> Data_5M { get; set; } = new List<Candle>();
        public List<Candle> Data_15M { get; set; } = new List<Candle>();
    }
}