module q_update_single_agent #(
    parameter signed [15:0] ALPHA = 16'h001A,   // Q8.8 (approx 0.1)
    parameter signed [15:0] GAMMA = 16'h00F3    // Q8.8 (approx 0.95)
)(
    input clk,
    input reset,

    input  [5:0] s,            // 64 states (6 bits)
    input  [1:0] a,            // 4 actions (2 bits)
    input  [5:0] s_next,       // Next state
    input  signed [15:0] reward, 
    input  update_en,

    output signed [15:0] new_Q_value,
    output new_Q_valid
);

    // Q-table: 64 States × 4 Actions
    reg signed [15:0] Q [0:63][0:3];

    // ---------------- Pipeline Stages ----------------
    // Stage 1: Capture inputs
    reg [5:0] s_s1, s_next_s1;
    reg [1:0] a_s1;
    reg signed [15:0] reward_s1;
    reg update_en_s1;

    // Stage 2: Read Q-values and compute max
    reg [5:0] s_s2;
    reg [1:0] a_s2;
    reg signed [15:0] oldQ_s2;
    reg signed [15:0] maxQnext_s2;
    reg signed [15:0] reward_s2;
    reg update_en_s2;

    // Stage 3: Compute new Q-value
    reg [5:0] s_s3;
    reg [1:0] a_s3;
    reg signed [15:0] newQ_s3;
    reg update_en_s3;

    // Stage 4: Writeback
    reg new_Q_valid_reg;

    // Forwarding logic
    reg [5:0] wb_s;
    reg [1:0] wb_a;
    reg signed [15:0] wb_newQ;
    reg wb_valid;

    integer st, ac;

    // ---------------- Reset / Init ----------------
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Initialize Q-table to zero
            for (st = 0; st < 64; st = st + 1) begin
                for (ac = 0; ac < 4; ac = ac + 1) begin
                    Q[st][ac] <= 16'sh0000;
                end
            end

            // Clear pipeline
            s_s1 <= 0; a_s1 <= 0; s_next_s1 <= 0; reward_s1 <= 0; update_en_s1 <= 0;
            s_s2 <= 0; a_s2 <= 0; oldQ_s2 <= 0; maxQnext_s2 <= 0; reward_s2 <= 0; update_en_s2 <= 0;
            s_s3 <= 0; a_s3 <= 0; newQ_s3 <= 0; update_en_s3 <= 0;
            new_Q_valid_reg <= 0;
            wb_s <= 0; wb_a <= 0; wb_newQ <= 0; wb_valid <= 0;
        end
        else begin
            // -------- Stage 1: Capture inputs --------
            s_s1         <= s;
            a_s1         <= a;
            s_next_s1    <= s_next;
            reward_s1    <= reward;
            update_en_s1 <= update_en;

            // -------- Stage 2: Read Q-values --------
            s_s2         <= s_s1;
            a_s2         <= a_s1;
            reward_s2    <= reward_s1;
            update_en_s2 <= update_en_s1;
            
            // Read old Q with forwarding check
            oldQ_s2 <= (wb_valid && wb_s == s_s1 && wb_a == a_s1) ? 
                       wb_newQ : Q[s_s1][a_s1];
            
            // Compute max Q(s_next) with forwarding
            maxQnext_s2 <= compute_max_q(s_next_s1);

            // -------- Stage 3: Compute new Q-value --------
            s_s3         <= s_s2;
            a_s3         <= a_s2;
            update_en_s3 <= update_en_s2;
            newQ_s3      <= compute_new_q(oldQ_s2, maxQnext_s2, reward_s2);

            new_Q_valid_reg <= update_en_s2;

            // -------- Stage 4: Writeback --------
            if (update_en_s3) begin
                Q[s_s3][a_s3] <= newQ_s3;
            end

            // Update forwarding info
            wb_s     <= s_s3;
            wb_a     <= a_s3;
            wb_newQ  <= newQ_s3;
            wb_valid <= update_en_s3;
        end
    end

    // Outputs
    assign new_Q_value = newQ_s3;
    assign new_Q_valid = new_Q_valid_reg;

    // ------------------ Helper Functions -------------------
    
    // Compute max Q(s_next, a') with forwarding
    function signed [15:0] compute_max_q;
        input [5:0] st;
        reg signed [15:0] q0, q1, q2, q3, m;
        begin
            // Check forwarding for each action
            q0 = (wb_valid && wb_s == st && wb_a == 2'd0) ? wb_newQ : Q[st][0];
            q1 = (wb_valid && wb_s == st && wb_a == 2'd1) ? wb_newQ : Q[st][1];
            q2 = (wb_valid && wb_s == st && wb_a == 2'd2) ? wb_newQ : Q[st][2];
            q3 = (wb_valid && wb_s == st && wb_a == 2'd3) ? wb_newQ : Q[st][3];
            
            m = q0;
            if (q1 > m) m = q1;
            if (q2 > m) m = q2;
            if (q3 > m) m = q3;
            compute_max_q = m;
        end
    endfunction

    // Q-learning update: Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    function signed [15:0] compute_new_q;
        input signed [15:0] oldQ;
        input signed [15:0] maxQnext;
        input signed [15:0] rew;
        reg signed [31:0] gamma_mul;
        reg signed [31:0] alpha_mul;
        reg signed [15:0] target;
        reg signed [15:0] td_error;
        begin
            gamma_mul = maxQnext * GAMMA;        // Q16.16
            target = rew + (gamma_mul >>> 8);    // r + γ·max Q(s',a')
            td_error = target - oldQ;            // TD error
            alpha_mul = td_error * ALPHA;        // α·TD_error
            compute_new_q = oldQ + (alpha_mul >>> 8); // New Q-value
        end
    endfunction

endmodule
