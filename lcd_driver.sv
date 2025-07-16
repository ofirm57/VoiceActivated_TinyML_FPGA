module lcd_driver(
         
    input logic clk ,     
    input logic rstb ,
    input logic  [7:0] char_in ,  
    input logic  char_valid,
    
    output logic [7:0]  lcd_data,
    output logic  lcd_rs,
    output logic  lcd_rw,
    output logic  lcd_en,
    output logic  char_ready
);

parameter integer WAIT_PARAMETER = 2000;
logic [15:0] wait_counter;

typedef enum logic [1:0] {
    IDLE,
    SETUP,
    PULSE_EN,
    WAIT
} lcd_driver_state_t;

lcd_driver_state_t cs, ns;


always_ff @(posedge clk or negedge rstb) begin
    if (~rstb) begin
        cs <= IDLE;
        wait_counter <= 15'd0;
    end else begin
        cs <= ns;
        if (cs == WAIT)
            wait_counter <= wait_counter + 1;
        else
            wait_counter <= 0;
    end
end


always_comb begin
    case (cs)
        IDLE:      ns = char_valid ? SETUP : IDLE;
        SETUP:     ns = PULSE_EN;
        PULSE_EN:  ns = WAIT;
        WAIT:      ns = (wait_counter == WAIT_PARAMETER) ? IDLE : WAIT ;
        default    ns = cs; 
    endcase
end

always_ff @(posedge clk or negedge rstb) begin
    if (~rstb) begin
        lcd_en <= 0;
        lcd_rs <= 0;
        lcd_rw <= 0;
        lcd_data <= 0;
        char_ready <= 1;
    end else begin
        case (cs)
            IDLE: begin
                char_ready <= 1;
                lcd_en <= 0;
            end
            SETUP: begin
                char_ready <= 0;
                lcd_data <= char_in;
                lcd_rs <= 1; // Data
                lcd_rw <= 0; // Write
            end
            PULSE_EN: begin
                lcd_en <= 1;
            end
            WAIT: begin
                lcd_en <= 0;
            end

        endcase
    end
end

endmodule //lcd_driver