using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{

    public GameObject paddle;
    public GameObject ball;
    private Rigidbody2D brb;
    private float yVelocity;
    private float paddleMinY = 8.8f;
    private float paddleMaxY = 17.4f;
    private float paddleMaxSpeed = 15;
    public float numSaved = 0;
    public float numMissed = 0;

    private ANN ann;

    private void Start()
    {
        ann = new ANN(6, 1, 1, 4, 0.11);
        brb = ball.GetComponent<Rigidbody2D>();
    }

    // pass in ball X and Y, ball Velocity X and Y, paddle X and Y, and paddle Velocity
    List<double> Run(double bx, double by, double bvx, double bvy, double px, double py, double pv, bool train)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(bx);
        inputs.Add(by);
        inputs.Add(bvx);
        inputs.Add(bvy);
        inputs.Add(px);
        inputs.Add(py);
        outputs.Add(pv);

        if (train)
            return (ann.Train(inputs, outputs));
        else
            return (ann.CalcOutput(inputs, outputs));

    }

    private void Update()
    {
        // update will provide all the actions for the paddle

        // posy is the position the paddle will move to, based on the yVelocity but also clamped
        // yVelocity is the output value of the neural network
        float posy = Mathf.Clamp(paddle.transform.position.y + (yVelocity * Time.deltaTime * paddleMaxSpeed),
                                paddleMinY, paddleMaxY);
        paddle.transform.position = new Vector3(paddle.transform.position.x, posy, paddle.transform.position.z);
        List<double> output = new List<double>();
        int layerMask = 1 << 6; // 1 bit shift 9 selects the back wall layer mask
        // create a raycast along the balls velocity
        RaycastHit2D hit = Physics2D.Raycast(ball.transform.position, brb.velocity, 1000, layerMask);

        // if we hit the top or bottom, calulate also the reflection angle for the paddle
        if (hit.collider != null && hit.collider.gameObject.tag == "tops")
        {
            Vector3 reflection = Vector3.Reflect(brb.velocity, hit.normal);
            hit = Physics2D.Raycast(hit.point, reflection, 1000, layerMask);
        }


        // if the ball has a clear shot to the back wall, the paddle Y position must be adjusted
        if (hit.collider != null && hit.collider.gameObject.tag == "backwall")
        {
            // delta Y - the change in Y between our hit and the paddle
            float dy = (hit.point.y - paddle.transform.position.y);
            // train neural network with current parameters and delta y as target
            output = Run(ball.transform.position.x,
                        ball.transform.position.y,
                        brb.velocity.x, brb.velocity.y,
                        paddle.transform.position.x,
                        paddle.transform.position.y,
                        dy, true);
            // set y velocity to output
            yVelocity = (float)output[0];
        }
        else
            yVelocity = 0;
    }

}





