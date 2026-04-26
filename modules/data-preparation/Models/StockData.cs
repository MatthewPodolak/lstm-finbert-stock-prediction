using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LSTM_DATA_SHAPER.Models
{
    public class StockData
    {
        public string? FolderPath { get; set; }
        public string Ticker { get; set; } = "Unknown.US";
        public List<Candle> Data { get; set; } = new List<Candle>();
    }
}